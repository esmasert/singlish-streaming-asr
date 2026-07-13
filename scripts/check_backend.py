from __future__ import annotations

import json
import os
import urllib.request

ws_url = os.getenv("NEMOTRON_WS_URL", "ws://127.0.0.1:8011/v1/ws/transcribe")
health_url = ws_url.replace("ws://", "http://").replace("wss://", "https://").replace("/v1/ws/transcribe", "/health")
with urllib.request.urlopen(health_url, timeout=10) as response:
    print(json.dumps(json.load(response), indent=2))
