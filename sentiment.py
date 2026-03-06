"""
센티멘트 지표 알고리즘
- VADER + 크립토 도메인 사전 보정
- 영향력 가중 평균 (log 스케일)
- 소스별 가중치 합산 → 최종 점수 (-100 ~ +100)
"""
import re
import math
from datetime import datetime, timezone

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from collector import TwitterCollector, RedditCollector, NewsCollector


# ── 설정 ────────────────────────────────────────────────────

COINS = {
    "BTC": ["bitcoin", "BTC", "$BTC"],
    "ETH": ["ethereum", "ETH", "$ETH"],
    "SOL": ["solana",   "SOL", "$SOL"],
}

SOURCE_WEIGHTS = {
    "twitter": 0.40,
    "reddit":  0.35,
    "news":    0.25,
}

LABELS = [
    ( 60,  100, "Extreme Greed", "극도의 탐욕"),
    ( 20,   59, "Greed",         "탐욕"),
    (-19,   19, "Neutral",       "중립"),
    (-59,  -20, "Fear",          "공포"),
    (-100, -60, "Extreme Fear",  "극도의 공포"),
]

CRYPTO_LEXICON = {
    "moon": 2.5, "mooning": 2.5, "moonshot": 2.5,
    "bullish": 2.0, "bull": 1.5, "pump": 1.0,
    "hodl": 1.5, "wagmi": 2.0, "buidl": 1.5,
    "bearish": -2.0, "bear": -1.5, "dump": -1.5,
    "crash": -2.5, "scam": -3.0, "fud": -1.5,
    "rekt": -2.5, "ngmi": -2.0, "rug": -3.0,
    "ponzi": -3.0, "dead": -2.0, "worthless": -2.5,
    "sell": 0.0, "buy": 0.0, "trade": 0.0,
}


# ── 텍스트 전처리 ────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+",     "", text)
    text = re.sub(r"@\w+",        "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s$]",    "", text)
    return text.strip()


# ── 감성 엔진 ────────────────────────────────────────────────

class SentimentEngine:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(CRYPTO_LEXICON)

    def score(self, text: str) -> float:
        return self.vader.polarity_scores(clean_text(text))["compound"]


# ── 지표 계산 ────────────────────────────────────────────────

class SentimentIndicator:

    def __init__(self, twitter_token: str, reddit_id: str,
                 reddit_secret: str, news_api_key: str = ""):
        self.engine  = SentimentEngine()
        self.twitter = TwitterCollector(twitter_token, COINS)
        self.reddit  = RedditCollector(reddit_id, reddit_secret, COINS)
        self.news    = NewsCollector(news_api_key, COINS)

    def _weighted_score(self, items: list[dict]) -> float | None:
        if not items:
            return None
        ws, wt = 0.0, 0.0
        for item in items:
            s = self.engine.score(item["text"])
            w = math.log1p(item["influence"])
            ws += s * w
            wt += w
        return ws / wt if wt else None

    def calculate(self, coin: str) -> dict:
        print(f"\n[{coin}] 수집 중...")

        raw = {
            "twitter": self.twitter.collect(coin),
            "reddit":  self.reddit.collect(coin),
            "news":    self.news.collect(coin),
        }

        source_scores = {}
        for src, items in raw.items():
            sc = self._weighted_score(items)
            source_scores[src] = sc
            print(f"  {src:8s}: {len(items):4d}건  score={sc:.4f}" if sc is not None
                  else f"  {src:8s}: 데이터 없음")

        valid        = {s: v for s, v in source_scores.items() if v is not None}
        if not valid:
            return {"coin": coin, "score": 0.0, "label_en": "No Data", "label_ko": "데이터 없음"}

        total_w      = sum(SOURCE_WEIGHTS[s] for s in valid)
        final_raw    = sum(SOURCE_WEIGHTS[s] * v for s, v in valid.items()) / total_w
        final_score  = round(final_raw * 100, 2)

        label_en, label_ko = "Neutral", "중립"
        for lo, hi, en, ko in LABELS:
            if lo <= final_score <= hi:
                label_en, label_ko = en, ko
                break

        return {
            "coin":          coin,
            "score":         final_score,
            "label_en":      label_en,
            "label_ko":      label_ko,
            "source_scores": {k: round(v * 100, 2) for k, v in valid.items()},
            "data_counts":   {k: len(v) for k, v in raw.items()},
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    def calculate_all(self) -> pd.DataFrame:
        return pd.DataFrame([self.calculate(c) for c in COINS])


# ── 시각화 ──────────────────────────────────────────────────

def plot_sentiment(df: pd.DataFrame):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("pip install plotly")
        return

    color_map = {
        "Extreme Greed": "#00C853", "Greed": "#AEEA00",
        "Neutral": "#90A4AE",
        "Fear": "#FF6D00",          "Extreme Fear": "#D50000",
    }

    fig = make_subplots(rows=1, cols=len(df),
                        subplot_titles=list(df["coin"]))
    for i, row in df.iterrows():
        fig.add_trace(go.Indicator(
            mode  = "gauge+number",
            value = row["score"],
            title = {"text": row["label_ko"]},
            gauge = {
                "axis":  {"range": [-100, 100]},
                "bar":   {"color": color_map.get(row["label_en"], "#90A4AE")},
                "steps": [
                    {"range": [-100, -60], "color": "#FFCDD2"},
                    {"range": [ -60, -20], "color": "#FFE0B2"},
                    {"range": [ -20,  20], "color": "#F5F5F5"},
                    {"range": [  20,  60], "color": "#F9FBE7"},
                    {"range": [  60, 100], "color": "#E8F5E9"},
                ],
            },
        ), row=1, col=i + 1)

    fig.update_layout(title_text="Crypto Sentiment Dashboard", height=400)
    fig.show()
