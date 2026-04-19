import json
from pathlib import Path

from modelrunner.sdk import ModelRunnerClient

_payload = Path(__file__).parent / "wkmeans" / "payload1.json"

with open(_payload, encoding="utf-8") as f:
    features = json.load(f)

client = ModelRunnerClient(host="localhost", port=8000)
response = client.predict(features=features)

print(response)

client.close()
