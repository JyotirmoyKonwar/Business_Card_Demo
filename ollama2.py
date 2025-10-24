import requests
import json
import os

# Path to your business card image
image_path = "WhatsApp Image 2025-10-22 at 22.56.56.jpeg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

# Ollama API endpoint (make sure `ollama serve` is running)
url = "http://localhost:11434/api/generate"

# Define the prompt directly for generation
prompt = (
    "You are an expert assistant that extracts contact information from business card images. "
    "Analyze the provided image and return *only* a single, valid JSON object with the following fields:\n"
    "{\n"
    '  "name": "...",\n'
    '  "title": "...",\n'
    '  "company": "...",\n'
    '  "phone": "...",\n'
    '  "email": "...",\n'
    '  "website": "...",\n'
    '  "address": "...",\n'
    '  "miscellaneous": "..."\n'
    "}\n\n"
    "If a field is missing, use null. Do not include any explanation or markdown — only valid JSON."
)

# Construct the payload for Ollama's generate endpoint
payload = {
    "model": "qwen2.5vl:3b-q4_K_M",
    "prompt": prompt,
    "images": [os.path.abspath(image_path)],  # attach image directly
    "stream": False  # disable streaming for easier handling
}

response = requests.post(url, json=payload)

if response.status_code != 200:
    print("❌ Request failed:", response.text)
    exit()

data = response.json()

# Extract model output
output_text = data.get("response", "").strip()

print("\n=== MODEL OUTPUT ===\n")
print(output_text)

# Try to parse and pretty print JSON if possible
try:
    parsed = json.loads(output_text)
    print("\n=== PARSED JSON ===\n")
    print(json.dumps(parsed, indent=2))
except Exception:
    print("\n⚠️ Could not parse JSON, raw output above.")
