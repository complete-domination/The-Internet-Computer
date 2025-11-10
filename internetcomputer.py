# -*- coding: utf-8 -*-
import os
import logging
import asyncio
from typing import Optional, List, Optional as Opt

import discord
from discord import app_commands
from discord.ext import commands
from openai import OpenAI

# ---------------- Configuration ----------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Optional usage line (non-streaming usage data is not guaranteed during stream)
SHOW_USAGE = os.environ.get("SHOW_USAGE", "0") == "1"

# Smooth typing controls
CHAR_RATE_MS = int(os.environ.get("CHAR_RATE_MS", "140"))       # how often to update the message
CHARS_PER_TICK = int(os.environ.get("CHARS_PER_TICK", "40"))    # how many characters to reveal per tick

GUILD_ID_RAW = os.environ.get("GUILD_ID")  # set to your server ID for instant sync
GUILD_ID: Optional[int] = int(GUILD_ID_RAW) if GUILD_ID_RAW and GUILD_ID_RAW.isdigit() else None

# Discord/Embed limits
DISCORD_LIMIT = 2000
EMBED_FIELD_LIMIT = 1024               # best practice per embed field
ANSWER_TOTAL_LIMIT = 5500              # keep headroom below overall embed cap

if not DISCORD_TOKEN:
    raise SystemExit("Missing DISCORD_TOKEN")
if not DEEPSEEK_API_KEY:
    raise SystemExit("Missing DEEPSEEK_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deepseek-bot")

# ---------------- Discord setup ----------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

# ---------------- DeepSeek client ----------------
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Answer clearly. "
    "If the user asks for code, provide minimal runnable examples. "
    "Keep answers under 6 paragraphs unless asked for more detail."
)

# ---------------- Helpers ----------------
def chunk_text(s: str, limit: int) -> List[str]:
    """Split text into chunks <= limit, on line boundaries when possible."""
    if len(s) <= limit:
        return [s]
    chunks, buf = [], []
    total = 0
    for line in s.splitlines(keepends=True):
        if total + len(line) > limit and buf:
            chunks.append("".join(buf))
            buf, total = [], 0
        if len(line) > limit:
            for i in range(0, len(line), limit):
                piece = line[i:i+limit]
                if total + len(piece) > limit and buf:
                    chunks.append("".join(buf))
                    buf, total = [], 0
                buf.append(piece)
                total += len(piece)
        else:
            buf.append(line)
            total += len(line)
    if buf:
        chunks.append("".join(buf))
    return [c.strip("\n") for c in chunks if c]

def make_embed(question: str, answer_preview: str = "⏳ generating...") -> discord.Embed:
    emb = discord.Embed(title="AI Response", color=discord.Color.blurple())
    for idx, qchunk in enumerate(chunk_text(question, EMBED_FIELD_LIMIT)):
        name = "Question" if idx == 0 else f"Question (cont. {idx})"
        emb.add_field(name=name, value=qchunk or "—", inline=False)
    emb.add_field(name="Answer", value=answer_preview, inline=False)
    return emb

def update_embed_with_answer(emb: discord.Embed, answer: str, usage_text: Opt[str]) -> discord.Embed:
    # Trim total answer for embed safety
    if len(answer) > ANSWER_TOTAL_LIMIT:
        answer = answer[:ANSWER_TOTAL_LIMIT].rstrip() + "\n…(truncated)"

    # Keep non-answer fields
    new_fields = []
    for f in emb.fields:
        if f.name.lower().startswith("answer"):
            continue
        new_fields.append((f.name, f.value, f.inline))

    # Add answer chunks
    achunks = chunk_text(answer, EMBED_FIELD_LIMIT)
    for i, chunk in enumerate(achunks):
        name = "Answer" if i == 0 else f"Answer (cont. {i})"
        new_fields.append((name, chunk if chunk else "—", False))

    if usage_text:
        new_fields.append(("Usage", usage_text[:EMBED_FIELD_LIMIT], False))

    # Rebuild embed (green = finished)
    new_emb = discord.Embed(title=emb.title, color=discord.Color.green())
    for name, value, inline in new_fields:
        new_emb.add_field(name=name, value=value, inline=inline)
    return new_emb

def update_embed_with_partial(emb: discord.Embed, partial: str, show_cursor: bool = True) -> discord.Embed:
    """Update only the answer placeholder with a partial (streaming) value."""
    # Trim for safety during streaming
    if len(partial) > ANSWER_TOTAL_LIMIT:
        partial = partial[:ANSWER_TOTAL_LIMIT].rstrip() + "\n…(truncated)"
    if show_cursor:
        partial = (partial + " ▋").rstrip()

    # Copy non-answer fields
    new_fields = []
    for f in emb.fields:
        if f.name.lower().startswith("answer"):
            continue
        new_fields.append((f.name, f.value, f.inline))

    # Add partial as answer fields
    achunks = chunk_text(partial, EMBED_FIELD_LIMIT)
    for i, chunk in enumerate(achunks):
        name = "Answer (streaming)" if i == 0 else f"Answer (cont. {i})"
        new_fields.append((name, chunk if chunk else "—", False))

    new_emb = discord.Embed(title=emb.title, color=discord.Color.blurple())
    for name, value, inline in new_fields:
        new_emb.add_field(name=name, value=value, inline=inline)
    return new_emb

async def send_initial_message(handle, embed: discord.Embed) -> discord.Message:
    """Send a public message for either an Interaction or a Context and return it."""
    if isinstance(handle, discord.Interaction):
        if not handle.response.is_done():
            await handle.response.defer(thinking=False, ephemeral=False)
        return await handle.followup.send(embed=embed)
    else:
        return await handle.send(embed=embed)

def get_channel(obj) -> Optional[discord.abc.Messageable]:
    if isinstance(obj, discord.Interaction):
        return obj.channel
    try:
        return obj.channel
    except AttributeError:
        return None

async def smooth_typing_display(
    msg: discord.Message,
    base_embed: discord.Embed,
    buffer: List[str],
    stop_event: asyncio.Event,
) -> None:
    """
    Periodically reveals small slices of the growing buffer, creating a typing look.
    Edits only the answer field (embed), keeping everything else stable.
    """
    shown_len = 0
    # Native typing indicator refresher (optional)
    async def keep_typing(ch):
        while not stop_event.is_set():
            try:
                await ch.trigger_typing()
            except Exception:
                pass
            await asyncio.sleep(7)

    typing_task = None
    channel = msg.channel
    if channel is not None:
        typing_task = asyncio.create_task(keep_typing(channel))

    try:
        while not stop_event.is_set():
            current = "".join(buffer)
            target_len = min(len(current), shown_len + CHARS_PER_TICK)
            if target_len > shown_len:
                # Human-like micro-pause on punctuation
                if target_len < len(current):
                    peek = current[target_len - 1:target_len]
                    if peek in ".?!,;:":
                        await asyncio.sleep(0.08)

                partial = current[:target_len]
                emb = update_embed_with_partial(base_embed, partial, show_cursor=True)
                try:
                    await msg.edit(embed=emb)
                except Exception:
                    pass
                shown_len = target_len

            await asyncio.sleep(CHAR_RATE_MS / 1000.0)

        # Final sync (cursor removed)
        final_text = "".join(buffer)
        emb = update_embed_with_partial(base_embed, final_text, show_cursor=False)
        try:
            await msg.edit(embed=emb)
        except Exception:
            pass
    finally:
        if typing_task:
            typing_task.cancel()

# ---------------- Response handler (smooth streaming) ----------------
async def respond_with_ai(interaction_or_ctx, question: str) -> None:
    """Posts the question immediately (embed), streams with smooth typing, then finalizes."""
    msg: discord.Message = None
    try:
        # 1) Post initial embed with "generating..."
        base_embed = make_embed(question)
        msg = await send_initial_message(interaction_or_ctx, base_embed)

        # 2) Prepare smooth display loop
        buffer: List[str] = []
        stop_event = asyncio.Event()
        display_task = asyncio.create_task(smooth_typing_display(msg, base_embed, buffer, stop_event))

        # 3) Stream from DeepSeek (ingest quickly; display loop handles pacing)
        try:
            stream = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                stream=True,
            )

            # The OpenAI-compatible iterator is sync; bridge it to our async task
            def sync_iter():
                for c in stream:
                    yield c

            for chunk in sync_iter():
                try:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                except Exception:
                    delta = None
                if not delta:
                    continue
                buffer.append(delta)
        finally:
            # Stop the display loop and wait for final sync
            stop_event.set()
            await display_task

        # 4) Finalize with full answer (+ optional usage when available)
        final_answer = "".join(buffer).strip() or "No answer returned."
        usage_text = None  # Streaming responses generally don't include reliable usage
        final_embed = update_embed_with_answer(base_embed, final_answer, usage_text)
        await msg.edit(embed=final_embed)

    except Exception as e:
        log.exception("DeepSeek streaming failed")
        err_msg = f"⚠️ Error: {e}"
        try:
            if isinstance(interaction_or_ctx, discord.Interaction):
                if not interaction_or_ctx.response.is_done():
                    await interaction_or_ctx.response.send_message(err_msg, ephemeral=True)
                else:
                    await interaction_or_ctx.followup.send(err_msg, ephemeral=True)
            else:
                await interaction_or_ctx.send(err_msg)
        except Exception:
            pass

# ---------------- Slash Commands ----------------
@tree.command(name="ask", description="Ask the AI a question and get a reply.")
@app_commands.describe(question="Your question or prompt")
async def ask_slash(interaction: discord.Interaction, question: str) -> None:
    await respond_with_ai(interaction, question)

@tree.command(name="ping", description="Simple health check.")
async def ping_slash(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("pong 🏓", ephemeral=True)

# Legacy prefix command
@bot.command(name="ask")
async def ask_legacy(ctx: commands.Context, *, question: str) -> None:
    await respond_with_ai(ctx, question)

# Error handling for slash commands
@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    log.exception("App command error: %s", error)
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(f"⚠️ Command error: {error}", ephemeral=True)
        else:
            await interaction.followup.send(f"⚠️ Command error: {error}", ephemeral=True)
    except Exception:
        pass

# ---------------- On Ready ----------------
@bot.event
async def on_ready() -> None:
    try:
        # Sync globally and, if provided, to a specific guild for instant availability
        if GUILD_ID:
            guild_obj = discord.Object(id=GUILD_ID)
            tree.copy_global_to(guild=guild_obj)
            guild_synced = await tree.sync(guild=guild_obj)
            log.info("Guild slash commands synced to %s (%d).", GUILD_ID, len(guild_synced))
            log.info("Guild commands now: %s", [c.name for c in guild_synced])

        global_synced = await tree.sync()
        log.info("Global slash commands synced (%d).", len(global_synced))
        log.info("Global commands now: %s", [c.name for c in global_synced])

    except Exception as e:
        log.exception("Could not sync app commands: %s", e)

    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
