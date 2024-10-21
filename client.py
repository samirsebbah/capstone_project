import json
import requests

url = "http://localhost:8000/predict/"

payload_repo = "./payloads/"
# Read the JSON payload file into a dictionary object
with open(payload_repo + 'payload1.json', 'r') as file:
    payload = json.load(file)

response = requests.post(url, json=payload)
print(response.json())