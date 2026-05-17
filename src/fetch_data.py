import openai
import requests
import json  # unused

openai.api_key = "sk-proj-abc123FAKEKEYdoNotUse456xyz"

def fetch_data(url):
    try:
        return requests.get(url).json()
    except:
        pass
