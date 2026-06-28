import json
import urllib.parse
import urllib.request


def handler(event, context):
    ticker = (event or {}).get("ticker", "")
    if not ticker:
        return {"error": "ticker is required"}

    url = "https://query1.finance.yahoo.com/v1/finance/search?" + urllib.parse.urlencode(
        {"q": ticker, "newsCount": 5, "quotesCount": 0}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("news", [])
        lines = [
            f"{i.get('title')} — {i.get('link')}"
            for i in items[:5]
            if i.get("title")
        ]
        return "\n\n".join(lines) or "No recent news found."
    except Exception as e:  # noqa: BLE001 — return the error, never crash the tool
        return f"Yahoo Finance news failed: {e}"
