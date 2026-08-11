import requests

url = "http://127.0.0.1:8000/predict/latest"

response = requests.get(url, timeout=60)

print("Status code:", response.status_code)
print(response.json())

response.raise_for_status()
