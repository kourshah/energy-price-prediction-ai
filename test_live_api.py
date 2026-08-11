"""
Test the automatic endpoint while the API is running.
"""

import requests


URL = "http://127.0.0.1:8000/predict/latest"

response = requests.get(URL, timeout=120)

print("=" * 70)
print("AUTOMATIC LIVE PREDICTION TEST")
print("=" * 70)
print("Status code:", response.status_code)
print("Response:")
print(response.json())
print("=" * 70)

response.raise_for_status()
