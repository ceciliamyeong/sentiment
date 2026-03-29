from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CC_KEY = os.environ.get("CRYPTOCOMPARE_API_KEY")
LC_KEY = os.environ.get("LUNARCRUSH_API_KEY")

@app.get("/api/news")
async def get_news():
    if not CC_KEY:
        return {"error": "CRYPTOCOMPARE_API_KEY not set"}
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={CC_KEY}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10)
        return r.json()

@app.get("/api/coins")
async def get_coins():
    if not LC_KEY:
        return {"error": "LUNARCRUSH_API_KEY not set"}
    url = "https://lunarcrush.com/api4/public/coins/list/v2?sort=social_volume_24h&limit=20"
    headers = {"Authorization": f"Bearer {LC_KEY}"}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, timeout=10)
        return r.json()

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "cc_key": "set" if CC_KEY else "missing",
        "lc_key": "set" if LC_KEY else "missing"
    }
