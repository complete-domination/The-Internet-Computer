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
intents = discord.Intents.default
