"""Chat with a local model through the Ollama Python library (streams tokens as they arrive)."""
import ollama

messages = [{"role": "system", "content": "You are a friendly tutor. Keep answers short."}]

while True:
    question = input("you › ")
    if question in {"exit", "quit"}:
        break
    messages.append({"role": "user", "content": question})

    reply = ""
    for chunk in ollama.chat(model="qwen3:4b-instruct", messages=messages, stream=True):
        piece = chunk.message.content
        print(piece, end="", flush=True)
        reply += piece
    print()
    messages.append({"role": "assistant", "content": reply})  # memory: the model sees the whole chat
