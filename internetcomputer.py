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

SHOW_USAGE = os.environ.get("SHOW_USAGE", "0") == "1"

GUILD_ID_RAW = os.environ.get("GUILD_ID")
GUILD_ID: Optional[int] = int(GUILD_ID_RAW) if GUILD_ID_RAW and GUILD_ID_RAW.isdigit() else None

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

def chunk_text(s: str, limit: int) -> List[str]:
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
    # removed title="AI Response" only
    emb = discord.Embed(color=discord.Color.blurple())
    for idx, qchunk in enumerate(chunk_text(question, EMBED_FIELD_LIMIT)):
        name = "Question" if idx == 0 else f"Question (cont. {idx})"
        emb.add_field(name=name, value=qchunk or "—", inline=False)
    emb.add_field(name="Answer", value=answer_preview, inline=False)
    return emb

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
    new_emb = discord.Embed(color=discord.Color.green())  # no title
    for name, value, inline in new_fields:
        new_emb.add_field(name=name, value=value, inline=inline)
    return new_emb

def update_embed_with_partial(emb: discord.Embed, partial: str) -> discord.Embed:
    if len(partial) > ANSWER_TOTAL_LIMIT:
        partial = partial[:ANSWER_TOTAL_LIMIT].rstrip() + "\n…(truncated)"
    new_fields = []
    for f in emb.fields:
        if f.name.lower().startswith("answer"):
            continue
        new_fields.append((f.name, f.value, f.inline))
    achunks = chunk_text(partial, EMBED_FIELD_LIMIT)
    for i, chunk in enumerate(achunks):
        name = "Answer (streaming)" if i == 0 else f"Answer (cont. {i})"
        new_fields.append((name, chunk if chunk else "—", False))
    new_emb = discord.Embed(color=discord.Color.blurple())  # no title
    for name, value, inline in new_fields:
        new_emb.add_field(name=name, value=value, inline=inline)
    return new_emb

async def send_initial_message(handle, embed: discord.Embed) -> discord.Message:
    if isinstance(handle, discord.Interaction):
        if not handle.response.is_done():
            await handle.response.defer(thinking=False, ephemeral=False)
        return await handle.followup.send(embed=embed)
    else:
        return await handle.send(embed=embed)

async def respond_with_ai(interaction_or_ctx, question: str) -> None:
    msg: discord.Message = None
    try:
        base_embed = make_embed(question)
        msg = await send_initial_message(interaction_or_ctx, base_embed)
        stream = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            stream=True,
        )
        partial_buf: List[str] = []
        last_edit = 0.0
        edit_every_n_tokens = 30
        min_edit_interval = 0.25
        async def iterate_chunks():
            for chunk in stream:
                yield chunk
        async for chunk in iterate_chunks():
            try:
                delta = getattr(chunk.choices[0].delta, "content", None)
            except Exception:
                delta = None
            if not delta:
                continue
            partial_buf.append(delta)
            if len(partial_buf) % edit_every_n_tokens == 0:
                now = asyncio.get_event_loop().time()
                if now - last_edit >= min_edit_interval:
                    partial_text = "".join(partial_buf).strip()
                    emb = update_embed_with_partial(base_embed, partial_text)
                    await msg.edit(embed=emb)
                    last_edit = now
        final_answer = "".join(partial_buf).strip() or "No answer returned."
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

@tree.command(name="ask", description="Ask the AI a question and get a reply.")
@app_commands.describe(question="Your question or prompt")
async def ask_slash(interaction: discord.Interaction, question: str) -> None:
    await respond_with_ai(interaction, question)

@tree.command(name="ping", description="Simple health check.")
async def ping_slash(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("pong 🏓", ephemeral=True)

@bot.command(name="ask")
async def ask_legacy(ctx: commands.Context, *, question: str) -> None:
    await respond_with_ai(ctx, question)

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
