# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from openai import OpenAI

# ---------------- Config ----------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Set this to your server id for instant guild sync (recommended)
GUILD_ID_RAW = os.environ.get("GUILD_ID")
GUILD_ID: Optional[int] = int(GUILD_ID_RAW) if GUILD_ID_RAW and GUILD_ID_RAW.isdigit() else None

MAX_DISCORD_REPLY = 1800

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

# ---------------- DeepSeek client (OpenAI SDK) ----------------
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Answer clearly. "
    "If the user asks for code, provide minimal runnable examples. "
    "Keep answers under 6 paragraphs unless asked for more detail."
)

def clamp_discord(text: str) -> str:
    return text if len(text) <= MAX_DISCORD_REPLY else text[: MAX_DISCORD_REPLY - 20].rstrip() + "\n\n…(truncated)"

async def respond_with_ai(interaction_or_ctx, question: str) -> None:
    """Handles both slash interactions and legacy prefix messages."""
    thinking_msg = None

    if isinstance(interaction_or_ctx, discord.Interaction):
        # Always ack within 3s
        try:
            if not interaction_or_ctx.response.is_done():
                await interaction_or_ctx.response.defer(thinking=True)
        except Exception as e:
            log.warning("Defer failed: %s", e)
        try:
            thinking_msg = await interaction_or_ctx.followup.send("🤖 Thinking…")
        except Exception as e:
            log.warning("followup.send failed: %s", e)
    else:
        try:
            thinking_msg = await interaction_or_ctx.reply("🤖 Thinking…")
        except Exception as e:
            log.warning("ctx.reply failed: %s", e)

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            stream=False,
        )
        answer = resp.choices[0].message.content.strip()
        answer = clamp_discord(answer)
        if thinking_msg:
            await thinking_msg.edit(content=answer)
        else:
            if isinstance(interaction_or_ctx, discord.Interaction):
                await interaction_or_ctx.followup.send(answer)
            else:
                await interaction_or_ctx.send(answer)
    except Exception as e:
        err = f"⚠️ Error: {e}"
        log.exception("DeepSeek call failed")
        try:
            if thinking_msg:
                await thinking_msg.edit(content=err)
            else:
                if isinstance(interaction_or_ctx, discord.Interaction):
                    await interaction_or_ctx.followup.send(err, ephemeral=True)
                else:
                    await interaction_or_ctx.send(err)
        except Exception:
            pass

# ---------------- Slash commands (REGISTERED GLOBALLY) ----------------
@tree.command(name="ask", description="Ask the AI a question and get a reply.")
@app_commands.describe(question="Your question or prompt")
async def ask_slash(interaction: discord.Interaction, question: str) -> None:
    await respond_with_ai(interaction, question)

@tree.command(name="ping", description="Simple health check.")
async def ping_slash(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("pong 🏓", ephemeral=True)

# Legacy prefix command for backup
@bot.command(name="ask")
async def ask_legacy(ctx: commands.Context, *, question: str) -> None:
    await respond_with_ai(ctx, question)

# Surface app command errors nicely
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
        # 1) If a guild is provided, sync there first (instant)
        if GUILD_ID:
            guild_obj = discord.Object(id=GUILD_ID)
            # ensure global commands are copied to guild, then sync
            tree.copy_global_to(guild=guild_obj)
            guild_synced = await tree.sync(guild=guild_obj)
            log.info("Guild slash commands synced to %s (%d).", GUILD_ID, len(guild_synced))
            log.info("Guild commands now: %s", [c.name for c in guild_synced])

        # 2) Also sync global (helps if you use the bot elsewhere)
        global_synced = await tree.sync()
        log.info("Global slash commands synced (%d).", len(global_synced))
        log.info("Global commands now: %s", [c.name for c in global_synced])

    except Exception as e:
        log.exception("Could not sync app commands: %s", e)

    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
