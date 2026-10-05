"""Vision: a local multimodal model looks at an image file."""
import sys
import ollama

image = sys.argv[1] if len(sys.argv) > 1 else "images/thumbnail.jpg"
response = ollama.chat(
    model="gemma3:4b",
    messages=[{"role": "user", "content": "Describe this image in two sentences. What text can you read?", "images": [image]}],
)
print(response.message.content)
