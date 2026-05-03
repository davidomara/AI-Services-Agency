import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful AI business assistant."},
        {"role": "user", "content": "Give me 5 AI services I can sell to small businesses."},
    ],
)

print(response.choices[0].message.content)