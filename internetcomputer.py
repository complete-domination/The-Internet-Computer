# -*- coding: utf-8 -*-
import os
import logging
import asyncio
from typing import Optional, List, Optional as Opt, Tuple, Dict, Callable, Any

import discord
from discord import app_commands
from discord.ext import commands
from openai import OpenAI

# ---------------- Configuration ----------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

SHOW_USAGE = os.environ.get("SHOW_USAGE", "0") == "1"

# Smooth typing feel
CHAR_RATE_MS = int(os.environ.get("CHAR_RATE_MS", "140"))
CHARS_PER_TICK = int(os.environ.get("CHARS_PER_TICK", "40"))

GUILD_ID_RAW = os.environ.get("GUILD_ID")
GUILD_ID: Optional[int] = int(GUILD_ID_RAW) if GUILD_ID_RAW and GUILD_ID_RAW.isdigit() else None

# Discord limits
DISCORD_LIMIT = 2000
EMBED_FIELD_LIMIT = 1024
ANSWER_TOTAL_LIMIT = 5500

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
    if len(s) <= limit:
        return [s]
    chunks, buf = [], []
    total = 0
    for line in s.splitlines(keepends=True):
        if total + len(line) > limit and buf:
            chunks.append("".join(buf)); buf, total = [], 0
        if len(line) > limit:
            for i in range(0, len(line), limit):
                piece = line[i:i+limit]
                if total + len(piece) > limit and buf:
                    chunks.append("".join(buf)); buf, total = [], 0
                buf.append(piece); total += len(piece)
        else:
            buf.append(line); total += len(line)
    if buf:
        chunks.append("".join(buf))
    return [c.strip("\n") for c in chunks if c]

def get_asker_identity(src) -> Tuple[str, Optional[str]]:
    user = None
    if isinstance(src, discord.Interaction):
        user = src.user
    elif hasattr(src, "author"):
        user = getattr(src, "author", None)
    name = "Unknown User"
    avatar_url = None
    if user is not None:
        name = getattr(user, "display_name", None) or getattr(user, "name", "Unknown User")
        try:
            avatar_url = str(user.display_avatar.url)
        except Exception:
            avatar_url = None
    return name, avatar_url

def make_embed(question: str, asker_name: Optional[str], asker_avatar: Optional[str], answer_preview: str = "⏳ generating...") -> discord.Embed:
    # No title for a cleaner look
    emb = discord.Embed(color=discord.Color.blurple())
    if asker_name:
        try:
            emb.set_author(name=f"{asker_name} asked", icon_url=asker_avatar if asker_avatar else discord.Embed.Empty)
        except Exception:
            emb.set_author(name=f"{asker_name} asked")
    for idx, qchunk in enumerate(chunk_text(question, EMBED_FIELD_LIMIT)):
        name = "Question" if idx == 0 else f"Question (cont. {idx})"
        emb.add_field(name=name, value=qchunk or "—", inline=False)
    emb.add_field(name="Answer", value=answer_preview, inline=False)
    return emb

def _rebuild_embed_preserving_author(base: discord.Embed, color: discord.Color, fields: List[tuple]) -> discord.Embed:
    new_emb = discord.Embed(color=color)
    try:
        if base.author and base.author.name:
            new_emb.set_author(name=base.author.name, icon_url=base.author.icon_url)
    except Exception:
        pass
    for name, value, inline in fields:
        new_emb.add_field(name=name, value=value, inline=inline)
    return new_emb

def update_embed_with_answer(emb: discord.Embed, answer: str, usage_text: Opt[str]) -> discord.Embed:
    if len(answer) > ANSWER_TOTAL_LIMIT:
        answer = answer[:ANSWER_TOTAL_LIMIT].rstrip() + "\n…(truncated)"

    new_fields = []
    for f in emb.fields:
        if f.name.lower().startswith("answer"):
            continue
        new_fields.append((f.name, f.value, f.inline))

    achunks = chunk_text(answer, EMBED_FIELD_LIMIT)
    for i, chunk in enumerate(achunks):
        name = "Answer" if i == 0 else f"Answer (cont. {i})"
        new_fields.append((name, chunk if chunk else "—", False))

    if usage_text:
        new_fields.append(("Usage", usage_text[:EMBED_FIELD_LIMIT], False))

    return _rebuild_embed_preserving_author(emb, discord.Color.green(), new_fields)

def update_embed_with_partial(emb: discord.Embed, partial: str, show_cursor: bool = True) -> discord.Embed:
    if len(partial) > ANSWER_TOTAL_LIMIT:
        partial = partial[:ANSWER_TOTAL_LIMIT].rstrip() + "\n…(truncated)"
    if show_cursor:
        partial = (partial + " ▋").rstrip()

    new_fields = []
    for f in emb.fields:
        if f.name.lower().startswith("answer"):
            continue
        new_fields.append((f.name, f.value, f.inline))

    achunks = chunk_text(partial, EMBED_FIELD_LIMIT)
    for i, chunk in enumerate(achunks):
        name = "Answer (streaming)" if i == 0 else f"Answer (cont. {i})"
        new_fields.append((name, chunk if chunk else "—", False))

    return _rebuild_embed_preserving_author(emb, discord.Color.blurple(), new_fields)

async def send_initial_message(handle, embed: discord.Embed) -> discord.Message:
    if isinstance(handle, discord.Interaction):
        if not handle.response.is_done():
            await handle.response.defer(thinking=False, ephemeral=False)
        return await handle.followup.send(embed=embed)
    else:
        return await handle.send(embed=embed)

# ---------------- Smooth streaming machinery ----------------
SENTINEL = object()
STREAM_STALL_TIMEOUT = float(os.environ.get("STREAM_STALL_TIMEOUT", "25"))  # seconds without any delta -> stop gracefully

async def smooth_typing_display(
    msg: discord.Message,
    base_embed: discord.Embed,
    q: asyncio.Queue,
) -> str:
    """Consume text deltas from the queue and reveal them as smooth 'typing'. Returns final text."""
    assembled: List[str] = []
    shown_len = 0
    done = False
    last_new_char = asyncio.get_running_loop().time()

    async def keep_typing(ch):
        while not done:
            try:
                await ch.trigger_typing()
            except Exception:
                pass
            await asyncio.sleep(7)

    typing_task = asyncio.create_task(keep_typing(msg.channel)) if msg.channel else None

    try:
        while True:
            got_new = False
            # Drain queue quickly
            while True:
                try:
                    item = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is SENTINEL:
                    done = True
                else:
                    assembled.append(item)
                    got_new = True

            if got_new:
                last_new_char = asyncio.get_running_loop().time()

            # Watchdog: if no new chars for a long time but not done, stop gracefully
            if not done and (asyncio.get_running_loop().time() - last_new_char) > STREAM_STALL_TIMEOUT:
                log.warning("Stream stall timeout reached; finalizing with current buffer.")
                done = True

            current = "".join(assembled)
            target_len = min(len(current), shown_len + CHARS_PER_TICK)
            if target_len > shown_len:
                if target_len < len(current):
                    peek = current[target_len - 1:target_len]
                    if peek in ".?!,;:":
                        await asyncio.sleep(0.08)
                partial = current[:target_len]
                emb = update_embed_with_partial(base_embed, partial, show_cursor=not done)
                try:
                    await msg.edit(embed=emb)
                except Exception:
                    pass
                shown_len = target_len

            if done and shown_len >= len(current):
                return current

            await asyncio.sleep(CHAR_RATE_MS / 1000.0)
    finally:
        if typing_task:
            typing_task.cancel()

def _produce_stream_sync(model: str, system_prompt: str, question: str, put_func: Callable[[Any], None]) -> None:
    """
    Blocking producer that reads the DeepSeek stream and pushes deltas via put_func.
    Exception-safe: always emits SENTINEL on exit.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
        )
        for chunk in stream:
            try:
                # OpenAI-compatible chunk shape
                ch = chunk.choices[0]
                # some providers send role deltas; ignore except for content
                delta_content = getattr(getattr(ch, "delta", None), "content", None)
            except Exception:
                delta_content = None
            if delta_content:
                put_func(delta_content)
    except Exception as e:
        # Log and continue; display loop will finalize with what we have
        try:
            log.exception("Streaming producer failed early: %s", e)
        except Exception:
            pass
    finally:
        # Always signal completion so the consumer doesn't hang
        try:
            put_func(SENTINEL)
        except Exception:
            pass

# ---------------- Response handler (smooth streaming + avatar) ----------------
async def respond_with_ai(interaction_or_ctx, question: str) -> None:
    """Posts the question with user's avatar, streams the answer token-by-token, then finalizes."""
    try:
        # Identify asker and build initial embed (no title)
        asker_name, asker_avatar = get_asker_identity(interaction_or_ctx)
        base_embed = make_embed(question, asker_name, asker_avatar)

        # Post initial embed
        msg = await send_initial_message(interaction_or_ctx, base_embed)

        # Prepare queue & background producer
        q: asyncio.Queue = asyncio.Queue()

        def put_now(item):
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(q.put_nowait, item)

        producer_task = asyncio.create_task(
            asyncio.to_thread(_produce_stream_sync, DEEPSEEK_MODEL, SYSTEM_PROMPT, question, put_now)
        )

        # Smooth typed display
        final_text = await smooth_typing_display(msg, base_embed, q)

        # Ensure producer completion
        try:
            await producer_task
        except Exception as pe:
            log.exception("Producer thread failed: %s", pe)

        # Finalize (green)
        final_answer = final_text.strip() or "No answer returned."
        usage_text = None
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
