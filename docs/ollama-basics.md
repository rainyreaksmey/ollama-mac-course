# Ollama basics

Ollama is a free, open-source tool for running large language models on your own computer. You download a model once with ollama pull, then run it offline with ollama run.

Ollama runs a local server. By default it listens on 127.0.0.1, port 11434. The terminal, the Ollama app and your own code all talk to this same server.

By default a model stays in memory for 5 minutes after the last request, then Ollama unloads it to free memory. ollama ps shows what is loaded and until when. ollama stop unloads a model right away.

On macOS, downloaded models are stored in the folder ~/.ollama/models.
