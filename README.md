# Ollama on Mac: the course project

All the code from the RainyTech video **[Ollama Tutorial for Beginners (2026)](https://youtu.be/c7ehlyoQrXY)**, run for real on a Mac mini M4 with 16 GB. Local AI, no cloud, no API key.

📺 Video: https://youtu.be/c7ehlyoQrXY · Advanced follow-up: https://youtu.be/8V5n0-7I2Cw
📝 Written tutorial: https://dev.to/rainytechlab/run-ai-locally-on-your-mac-with-ollama-chat-tools-rag-and-a-custom-model-real-numbers-from-a-4ngf

| File | What it shows |
|---|---|
| `chat.py` | A streaming chat app with memory (official `ollama` Python package) |
| `openai_api.py` | The OpenAI Python SDK talking to Ollama on localhost |
| `tools.py` | Tool calling: the model calls real Python functions (disk and memory of your Mac) |
| `rag.py` | RAG: answers from the notes in `docs/` with `nomic-embed-text` |
| `vision.py` | A vision model (Gemma 3) describes an image |
| `Modelfile` | A custom model: Rainy Tutor (same weights, new system prompt and settings) |
| `finetune/` | Fine-tune Llama 3.2 1B with MLX, then run it in Ollama |

## Real results (Mac mini M4, 16 GB, Ollama 0.35.1)

| Test | Result |
|---|---|
| Qwen 3 · 4B instruct | 35.8 tokens/s, 100% GPU, 3.2 GB loaded |
| Llama 3.2 · 3B | 44.8 tokens/s |
| Tool calling | called both functions → 150 GB disk free, 61% of 16 GB memory free |
| RAG | top match 0.869 → "Apple M4 with 16 GB" (plain Llama didn't know) |
| MLX LoRA fine-tune | 300 steps in ~6 min; best validation loss at step 100 (1.382), overfitting by step 300 (1.811) |
| Fine-tuned model in Ollama | 36.7 tokens/s |

## Setup

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull qwen3:4b-instruct
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull nomic-embed-text

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python chat.py
.venv/bin/python openai_api.py
.venv/bin/python tools.py
.venv/bin/python rag.py "Which Mac does the studio use?"
.venv/bin/python vision.py images/thumbnail.jpg

ollama create rainy-tutor -f Modelfile
ollama run rainy-tutor "What is a variable?"
```

## Fine-tune with MLX and run it in Ollama

The base model and trained adapters are not in this repo (several GB). Recreate them:

```bash
source .venv/bin/activate   # mlx-lm (from requirements.txt) provides mlx_lm.* and hf
cd finetune

# 1. dataset: 120 Q&A examples made by a LOCAL model (already in data/)
python make_data.py

# 2. base model
hf download mlx-community/Llama-3.2-1B-Instruct-bf16 --local-dir models/llama-3.2-1b-instruct

# 3. train a LoRA adapter (checkpoint every 100 steps)
mlx_lm.lora --model models/llama-3.2-1b-instruct --train --data data --iters 300 \
  --batch-size 4 --num-layers 16 --learning-rate 1e-4 \
  --steps-per-report 25 --steps-per-eval 100 --save-every 100 --adapter-path adapters

# 4. use the step-100 checkpoint (lowest validation loss) and fuse it
mkdir -p adapters-100
cp adapters/0000100_adapters.safetensors adapters-100/adapters.safetensors
cp adapters/adapter_config.json adapters-100/
mlx_lm.fuse --model models/llama-3.2-1b-instruct --adapter-path adapters-100 --save-path models/rainy-llama-1b

# 5. import into Ollama (the Modelfile adds the Llama 3 chat TEMPLATE it needs)
ollama create rainy-llama -f Modelfile
ollama run rainy-llama "What is an API?"
```

Note: the `SYSTEM` header in `finetune/Modelfile` contains the date the model was trained. mlx-lm adds that header to every training example, so keep it matching your own training date.

## License

MIT. Made by Rainy Reaksmey for [RainyTech on YouTube](https://www.youtube.com/@RainyTechLab).
