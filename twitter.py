"""
Twitter/X API v2 수집 모듈
"""
import requests
from datetime import datetime, timezone


class TwitterCollector:
    URL = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, bearer_token: str, coins: dict):
        self.headers = {"Authorization": f"Bearer {bearer_token}"}
        self.coins = coins

    def collect(self, coin: str, max_results: int = 100) -> list[dict]:
        kw    = " OR ".join(self.coins[coin])
        query = f"({kw}) lang:en -is:retweet -is:reply"
        params = {
            "query":        query,
            "max_results":  min(max_results, 100),
            "tweet.fields": "created_at,public_metrics",
        }
        try:
            data = requests.get(self.URL, headers=self.headers,
                                params=params, timeout=10).json().get("data", [])
        except Exception as e:
            print(f"[Twitter] Error: {e}")
            return []

        results = []
        for t in data:
            m = t.get("public_metrics", {})
            influence = max(1, m.get("like_count", 0) + m.get("retweet_count", 0) * 2)
            results.append({
                "source":     "twitter",
                "text":       t["text"],
                "influence":  influence,
                "created_at": t.get("created_at", ""),
            })
        return results
