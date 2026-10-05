# Ollama settings

The default context length is 4096 tokens. Raise it with the environment variable OLLAMA_CONTEXT_LENGTH, with /set parameter num_ctx inside a chat, or with the num_ctx option in an API request. A longer context uses more memory.

To change where the server listens, set OLLAMA_HOST. On the Mac app, set variables with launchctl setenv and then restart Ollama.

OLLAMA_KV_CACHE_TYPE can be f16 (the default), q8_0 or q4_0. q8_0 uses about half the memory of f16 for the context cache.
