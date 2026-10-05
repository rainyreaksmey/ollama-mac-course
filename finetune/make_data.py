"""Build a small fine-tuning dataset with a LOCAL model (no cloud): beginner questions -> answers in the
Rainy Tech house style. Output: data/train.jsonl, data/valid.jsonl in the chat format mlx-lm expects."""
import json, random
import ollama

STYLE = ("Answer as Rainy, the Rainy Tech tutor. Format exactly:\n"
         "line 1: a one-sentence answer.\n"
         "line 2: 'Think of it like ' + one everyday analogy.\n"
         "line 3: '🌧️ Rainy tip: ' + one practical tip.\n"
         "No markdown, no extra lines.")

TOPICS = """API, REST API, JSON, HTTP status codes, localhost, port number, IP address, DNS, HTTPS, cookies,
JWT token, OAuth, password hashing, environment variables, terminal, shell command, PATH variable, Homebrew,
Python virtual environment, pip, npm, Git commit, Git branch, pull request, merge conflict, Docker container,
Docker image, cloud server, CPU, GPU, RAM, unified memory, SSD, cache, compiler, interpreter, variable,
function, loop, recursion, class and object, database, SQL query, primary key, index in a database, NoSQL,
backend, frontend, framework, library, open source, license, bug, debugging, unit test, CI pipeline,
large language model, token in an LLM, context window, prompt, temperature in an LLM, embedding,
vector database, RAG, fine-tuning, LoRA, quantization, model parameters, inference, GPU memory, Ollama,
MLX, Apple Silicon, neural network, training data, overfitting, hallucination, AI agent, tool calling,
system prompt, open-weight model, GGUF file, Hugging Face, Python, JavaScript, TypeScript, Rust, Swift,
async code, thread, process, operating system, kernel, file system, compression, encryption, VPN,
firewall, two-factor authentication, phishing, backup, version number, semantic versioning, YAML,
Markdown, regular expression, web socket, CDN, load balancer, latency, bandwidth, webhook, cron job,
CLI vs GUI, IDE, VS Code extension, keyboard shortcut, Screen recording codec, 4K resolution, frame rate""".replace("\n", " ")
topics = [t.strip() for t in TOPICS.split(",") if t.strip()]
FORMS = ["What is {}?", "Explain {} simply.", "Why does {} matter?", "Can you explain {} to a beginner?"]

rows = []
for i, topic in enumerate(topics):
    q = random.Random(i).choice(FORMS).format(topic)
    r = ollama.chat(model="qwen3:4b-instruct", messages=[{"role": "system", "content": STYLE}, {"role": "user", "content": q}],
                    options={"temperature": 0.5})
    lines = [l.strip() for l in r.message.content.strip().split("\n") if l.strip()]
    if len(lines) != 3 or not lines[1].startswith("Think of it like") or not lines[2].startswith("🌧️ Rainy tip:"):
        print("skip", q); continue
    rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "\n".join(lines)}]})
    print(len(rows), q)

random.Random(7).shuffle(rows)
n = max(8, len(rows) // 10)
for name, part in [("valid", rows[:n]), ("train", rows[n:])]:
    with open(f"data/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
print(f"train {len(rows) - n}  valid {n}")
