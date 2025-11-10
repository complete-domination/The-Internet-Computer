# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from openai import OpenAI

# ---------------- Configuration ----------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

GUILD_ID_RAW = os.environ.get("GUILD_ID")  # set to your server ID for instant sync
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

# ---------------- DeepSeek client ----------------
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Answer clearly. "
    "If the user asks for code, provide minimal runnable examples. "
    "Keep answers under 6 paragraphs unless asked for more detail."
)

def clamp_discord(text: str) -> str:
    """Avoid exceeding Discord's message length limit."""
    return text if len(text) <= MAX_DISCORD_REPLY else text[:MAX_DISCORD_REPLY - 20].rstrip() + "\n\n…(truncated)"

# ---------------- Response handler ----------------
async def respond_with_ai(interaction_or_ctx, question: str) -> None:
    """Posts the question first, then edits to include the answer."""
    try:
        header = f"**🧠 Question:** {question}"

        # Post the question publicly right away
        if isinstance(interaction_or_ctx, discord.Interaction):
            await interaction_or_ctx.response.defer(thinking=False)
            msg = await interaction_or_ctx.followup.send(header)
        else:
            msg = await interaction_or_ctx.send(header)

        # Call DeepSeek API
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

        # Edit the same message with the AI’s reply appended
        await msg.edit(content=f"{header}\n\n**💬 Answer:** {answer}")

    except Exception as e:
        err_msg = f"⚠️ Error: {e}"
        log.exception("DeepSeek call failed")
        try:
            if isinstance(interaction_or_ctx, discord.Interaction):
                await interaction_or_ctx.followup.send(err_msg)
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
        # Always sync globally, and also to the guild if specified
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
