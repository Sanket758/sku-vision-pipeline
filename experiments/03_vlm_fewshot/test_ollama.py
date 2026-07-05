import requests
import base64
import sys

def test_ollama():
    try:
        with open("../../Dataset/raw/kaufland/IMG20260601171129.jpg", "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "model": "moondream:latest",
            "prompt": "List all the products you can see on the shelf.",
            "images": [img_b64],
            "stream": False
        }
        print("Sending request to Ollama (moondream)...")
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        print("Response:")
        print(response.json()["response"])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ollama()
