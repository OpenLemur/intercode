import requests, json
import time
from rich import print
import re
import os
import config

# Set HuggingFace Endpoint URL from environment or config file
endpoint = os.environ.get("HF_API_URL")
if (endpoint is None or endpoint == "") and os.path.isfile(os.path.join(os.getcwd(), "keys.cfg")):
    cfg = config.Config('keys.cfg')
    endpoint = cfg.get("HF_API_URL")
assert(endpoint != None)
HF_API_URL = endpoint


def HFChat(message, stop_sequences=None):
    # Prevent RateLimit errors
    # time.sleep(1)
    payload = {"inputs": message, "parameters": {"max_new_tokens": 512, "temperature": 0.01, "stop": stop_sequences, "return_full_text": False}}
    response = requests.post(HF_API_URL, json=payload)
    
    if response.status_code == 422:
        print(response.json()["error"])
        if "Given: " in response.json()["error"]:
            error_msg = response.json()["error"]
            given_length = int(re.search(r"Given: (\d+) `inputs` tokens", error_msg).group(1))
            payload["parameters"]["max_new_tokens"] = 4096 - given_length - 1
            response = requests.post(HF_API_URL, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return ""
        
    output = response.json()
    result = output[0]["generated_text"].strip()
    return result