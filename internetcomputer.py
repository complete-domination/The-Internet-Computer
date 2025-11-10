import os
import asyncio
import logging
from typing import Optional, Tuple

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

# ---------------- Config ----------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
AI_BACKEND = (os.environ.get("AI_BACKEND") or "AUTO").upper()  # AUTO | HUGGINGFACE | OLLAMA

# Hugging Face settings
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"
HF_API_BASE = os.environ.get("HF_API_BASE") or "https://router.huggingface.co/hf-inference"  # new router

# Ollama settings (local server; not for Railway unless you host it)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or "llama3.2"

# Guild sync for instant slash commands
GUILD_ID_RAW = os.environ.get("GUILD_ID")
GUILD_ID: Optional[int] = int(GUILD_ID_RAW) if GUILD_ID_RAW and GUILD_ID_RAW.isdigit() else None

# Message/response settings
MAX_DISCORD_REPLY = 1800  # below 2000 hard limit
REQUEST_TIMEOUT = 45

if not DISCORD_TOKEN:
    raise SystemExit("Missing DISCORD_TOKEN")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qa-bot")

# ---------------- Discord setup ----------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

# ---------------- AI backends ----------------
SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Answer clearly. "
    "If the user asks for code, provide minimal runnable examples. "
    "Keep answers under 6 paragraphs unless asked for more detail."
)

async def call_huggingface(session: aiohttp.ClientSession, prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, error) using the new HF router endpoint."""
    if not HF_API_TOKEN:
        return None, "Hugging Face selected but HF_API_TOKEN is not set."

    model = HF_MODEL
    base = HF_API_BASE.rstrip("/")
    url = f"{base}/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    # Llama-style instruct formatting
    formatted = (
        f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n"
        f"<|user|>\n{prompt}\n</|user|>\n<|assistant|>\n"
    )

    payload = {
        "inputs": formatted,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "repetition_penalty": 1.05,
        },
        "options": {"wait_for_model": True},
    }

    try:
        async with session.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT) as resp:
            text = await resp.text()
            if resp.status != 200:
                # Helpful hint if someone still uses the old endpoint
                if resp.status == 410 or "no longer supported" in text.lower():
                    return None, "Hugging Face: old API endpoint detected; use router.huggingface.co/hf-inference."
                return None, f"Hugging Face error {resp.status}: {text[:300]}"

            data = await resp.json()
            # Typical responses:
            # 1) list of {generated_text: "..."}
            if isinstance(data, list) and data and "generated_text" in data[0]:
                full = data[0]["generated_text"]
                out = full.split("<|assistant|>")[-1].strip()
                return out, None
            # 2) dict with generated_text
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip(), None
            # 3) dict with error
            if isinstance(data, dict) and "error" in data:
                return None, f"Hugging Face: {data['error']}"
            return None, "Hugging Face: unexpected response format."
    except asyncio.TimeoutError:
        return None, "Hugging Face: request timed out."
    except Exception as e:
        return None, f"Hugging Face exception: {e}"

async def call_ollama(session: aiohttp.ClientSession, prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, error). Requires local Ollama running."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:",
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512},
    }
    try:
        async with session.post(url, json=payload, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status != 200:
                text = await resp.text()
                return None, f"Ollama error {resp.status}: {text[:300]}"
            data = await resp.json()
            out = data.get("response", "").strip()
            if not out:
                return None, "Ollama: empty response."
            return out, None
    except asyncio.TimeoutError:
        return None, "Ollama: request timed out."
    except Exception as e:
        return None, f"Ollama exception: {e}"

async def ask_ai(prompt: str) -> str:
    """Dispatch to selected backend and return answer text (or raise RuntimeError)."""
    async with aiohttp.ClientSession() as session:
        backend = AI_BACKEND
        if backend == "AUTO":
            backend = "HUGGINGFACE" if HF_API_TOKEN else "OLLAMA"

        if backend == "HUGGINGFACE":
            text, err = await call_huggingface(session, prompt)
            if err and AI_BACKEND == "AUTO":
                log.warning("HF failed (%s). Trying Ollama fallback.", err)
                text, err = await call_ollama(session, prompt)
            if err:
                raise RuntimeError(err)
            return text or "I couldn't produce a response."
        elif backend == "OLLAMA":
            text, err = await call_ollama(session, prompt)
            if err:
                raise RuntimeError(err)
            return text or "I couldn't produce a response."
        else:
            raise RuntimeError(f"Unknown AI_BACKEND: {AI_BACKEND}")

# ---------------- Helpers ----------------
def clamp_discord(text: str) -> str:
    if len(text) <= MAX_DISCORD_REPLY:
        return text
    return text[:MAX_DISCORD_REPLY - 20].rstrip() + "\n\n…(truncated)"

async def respond_with_ai(interaction_or_ctx, question: str):
    """Handles both slash command interactions and legacy ctx."""
    thinking_msg = None
    try:
        if isinstance(interaction_or_ctx, discord.Interaction):
            await interaction_or_ctx.response.defer(thinking=True)
            thinking_msg = await interaction_or_ctx.followup.send("🤖 Thinking…")
        else:
            thinking_msg = await interaction_or_ctx.reply("🤖 Thinking…")

        answer = await ask_ai(question)
        answer = clamp_discord(answer)
        if thinking_msg:
            await thinking_msg.edit(content=answer)
    except Exception as e:
        err_text = f"⚠️ Error: {e}"
        if thinking_msg:
            try:
                await thinking_msg.edit(content=err_text)
            except Exception:
                pass
        else:
            if isinstance(interaction_or_ctx, discord.Interaction):
                if not interaction_or_ctx.response.is_done():
                    await interaction_or_ctx.response.send_message(err_text, ephemeral=True)
                else:
                    await interaction_or_ctx.followup.send(err_text, ephemeral=True)
            else:
                await interaction_or_ctx.send(err_text)

# ---------------- Commands ----------------
def register_commands():
    # We define once, then optionally bind to a guild for instant availability
    @app_commands.describe(question="Your question or prompt")
    async def ask_cmd(interaction: discord.Interaction, question: str):
        await respond_with_ai(interaction, question)

    if GUILD_ID:
        guild_obj = discord.Object(id=GUILD_ID)
        tree.add_command(app_commands.Command(
            name="ask",
            description="Ask the AI a question and get a reply.",
            callback=ask_cmd
        ), guild=guild_obj)
    else:
        tree.add_command(app_commands.Command(
            name="ask",
            description="Ask the AI a question and get a reply.",
            callback=ask_cmd
        ))

@bot.command(name="ask")
async def ask_legacy(ctx: commands.Context, *, question: str):
    await respond_with_ai(ctx, question)

@bot.event
async def on_ready():
    try:
        register_commands()
        if GUILD_ID:
            guild_obj = discord.Object(id=GUILD_ID)
            tree.copy_global_to(guild=guild_obj)  # ensure any future globals copy over
            synced = await tree.sync(guild=guild_obj)
            log.info("Slash commands synced to guild %s (%d).", GUILD_ID, len(synced))
        else:
            synced = await tree.sync()
            log.info("Global slash commands synced (%d).", len(synced))
    except Exception as e:
        log.warning("Could not sync app commands: %s", e)

    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
