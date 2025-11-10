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
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # required if using HUGGINGFACE
HF_MODEL = os.environ.get("HF_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"

# Ollama settings (install from https://ollama.com/download and run `ollama run llama3.2`)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or "llama3.2"

# Message/response settings
MAX_DISCORD_REPLY = 1800  # keep under 2000 char hard limit; leave room for code fences
REQUEST_TIMEOUT = 45      # seconds for API calls

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
    """Return (text, error)."""
    if not HF_API_TOKEN:
        return None, "Hugging Face selected but HF_API_TOKEN is not set."
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    # Simple instruction-following format for Llama-style models
    formatted = f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n<|user|>\n{prompt}\n</|user|>\n<|assistant|>\n"

    payload = {
        "inputs": formatted,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "repetition_penalty": 1.05,
        },
        "options": {"wait_for_model": True}
    }
    try:
        async with session.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status != 200:
                text = await resp.text()
                return None, f"Hugging Face error {resp.status}: {text[:300]}"
            data = await resp.json()
            # HF returns a list with {'generated_text': '...'}
            if isinstance(data, list) and data and "generated_text" in data[0]:
                full = data[0]["generated_text"]
                # Extract assistant portion if present
                out = full.split("<|assistant|>")[-1].strip()
                return out, None
            # Some models return {'error': '...'}
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
        "options": {"temperature": 0.7, "num_predict": 512}
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
            backend = "HUGGINGFACE" if HF_API_TOKEN else "OLLLAMA"  # typo-proof next line
            backend = "HUGGINGFACE" if HF_API_TOKEN else "OLLAMA"

        if backend == "HUGGINGFACE":
            text, err = await call_huggingface(session, prompt)
            if err and AI_BACKEND == "AUTO":
                # fallback to Ollama automatically if available
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
            await interaction_or_ctx.response.defer(thinking=True)  # ephemeral thinking
            # send a followup message we can edit later
            thinking_msg = await interaction_or_ctx.followup.send("🤖 Thinking…")
        else:
            # commands.Context (prefix !ask)
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
@tree.command(name="ask", description="Ask the AI a question and get a reply.")
@app_commands.describe(question="Your question or prompt")
async def ask(interaction: discord.Interaction, question: str):
    await respond_with_ai(interaction, question)

@bot.command(name="ask")
async def ask_legacy(ctx: commands.Context, *, question: str):
    await respond_with_ai(ctx, question)


@bot.event
async def on_ready():
    try:
        await tree.sync()
        log.info("Slash commands synced.")
    except Exception as e:
        log.warning("Could not sync app commands: %s", e)
    log.info("Logged in as %s", bot.user)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
