"""Ollama speaks the OpenAI API too: the official OpenAI SDK, pointed at localhost."""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is required but unused

response = client.chat.completions.create(
    model="qwen3:4b-instruct",
    messages=[{"role": "user", "content": "In one sentence each: what is an API, and what is JSON?"}],
)
print(response.choices[0].message.content)
