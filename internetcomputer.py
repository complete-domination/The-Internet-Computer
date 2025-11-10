import os
import asyncio
import logging
from typing import Optional

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

# ---------------- Config ----------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

GUILD_ID_RAW = os.environ.get("GUILD_ID")  # set this for instant slash registration
GUILD_ID: Optional[int] = int(GUILD_ID_RAW) if GUILD_ID_RAW and GUILD_ID_RAW.isdigit() else None

MAX_DISCORD_REPLY = 1800
REQUEST_TIMEOUT = 45  # DeepSeek HTTP timeout (s)

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

# ---------------- DeepSeek call ----------------
SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Answer clearly. "
    "If the user asks for code, provide minimal runnable examples. "
    "Keep answers under 6 paragraphs unless asked for more detail."
)

async def call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"DeepSeek API error {resp.status}: {body[:300]}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except asyncio.TimeoutError:
            raise RuntimeError("DeepSeek request timed out.")
        except Exception as e:
            raise RuntimeError(f"DeepSeek exception: {e}")

# ---------------- Helpers ----------------
def clamp_discord(text: str) -> str:
    return text if len(text) <= MAX_DISCORD_REPLY else text[:MAX_DISCORD_REPLY - 20].rstrip() + "\n\n…(truncated)"

async def respond_with_ai(interaction_or_ctx, question: str):
    """Handles both slash interactions and legacy prefix messages."""
    thinking_msg = None

    # 1) Acknowledge within 3s
    if isinstance(interaction_or_ctx, discord.Interaction):
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

    # 2) Do the AI call
    try:
        answer = await call_deepseek(question)
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
        log.error("AI error: %s", e)
        try:
            if thinking_msg:
                await thinking_msg.edit(content=err)
            else:
                if isinstance(interaction_or_ctx, discord.Interaction):
                    await interaction_or_ctx.followup.send(err, ephemeral=True)
                else:
                    await interaction_or_ctx.send(err)
        except Exception as e2:
            log.error("Failed to send error message: %s", e2)

# ---------------- Slash commands (guild-bound if GUILD_ID provided) ----------------
if GUILD_ID:
    GUILD_OBJ = discord.Object(id=GUILD_ID)

    @tree.command(name="ask", description="Ask the AI a question and get a reply.", guild=GUILD_OBJ)
    @app_commands.describe(question="Your question or prompt")
    async def ask_slash(interaction: discord.Interaction, question: str):
        await respond_with_ai(interaction, question)

    @tree.command(name="ping", description="Simple health check.", guild=GUILD_OBJ)
    async def ping_slash(interaction: discord.Interaction):
        await interaction.response.send_message("pong 🏓", ephemeral=True)
else:
    @tree.command(name="ask", description="Ask the AI a question and get a reply.")
    @app_commands.describe(question="Your question or prompt")
    async def ask_slash(interaction: discord.Interaction, question: str):
        await respond_with_ai(interaction, question)

    @tree.command(name="ping", description="Simple health check.")
    async def ping_slash(interaction: discord.Interaction):
        await interaction.response.send_message("pong 🏓", ephemeral=True)

# Legacy prefix command
@bot.command(name="ask")
async def ask_legacy(ctx: commands.Context, *, question: str):
    await respond_with_ai(ctx, question)

# Global app command error surfacing
@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    log.exception("App command error: %s", error)
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(f"⚠️ Command error: {error}", ephemeral=True)
        else:
            await interaction.followup.send(f"⚠️ Command error: {error}", ephemeral=True)
    except Exception:
        pass

@bot.event
async def on_ready():
   try:
        if GUILD_ID:
            synced = await tree.sync(guild=discord.Object(id=GUILD_ID))
            log.info("Slash commands synced to guild %s (%d).", GUILD_ID, len(synced))
            log.info("Guild commands: %s", [c.name for c in synced])
        else:
            synced = await tree.sync()
            log.info("Global slash commands synced (%d).", len(synced))
            log.info("Global commands: %s", [c.name for c in synced])
    except Exception as e:
        log.exception("Could not sync app commands: %s", e)

    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
