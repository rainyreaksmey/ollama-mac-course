"""Tool calling: the model decides to call real Python functions that read this Mac's disk and memory."""
import inspect
import re
import subprocess
import ollama


def get_disk_free() -> str:
    """Get the free space on this Mac's main disk.

    Returns:
        Free space as text, for example "177 GB free of 228 GB".
    """
    out = subprocess.run(["df", "-g", "/"], capture_output=True, text=True).stdout.split("\n")[1].split()
    return f"{out[3]} GB free of {out[1]} GB"


def get_memory() -> str:
    """Get this Mac's total memory and how much is free right now.

    Returns:
        Memory as text, for example "16 GB total, 63% free".
    """
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    free = re.search(r"free percentage: (\d+)%", out).group(1)
    return f"{total} GB total, {free}% free"


TOOLS = {"get_disk_free": get_disk_free, "get_memory": get_memory}


def run_tool(name: str, args: dict, tools: dict = TOOLS) -> str:
    """Run one tool call. Unknown tools and missing or unexpected arguments come back as an error string."""
    fn = tools.get(name)
    if fn is None:
        return f"error: no tool named {name}"
    try:
        inspect.signature(fn).bind(**args)   # checks argument names only, not their types
    except TypeError as e:
        return f"error: bad arguments for {name}: {e}"
    return fn(**args)


if __name__ == "__main__":
    messages = [{"role": "user", "content": "How much free disk space and memory does my Mac have right now? Answer in one short sentence."}]

    response = ollama.chat(model="qwen3:4b-instruct", messages=messages, tools=list(TOOLS.values()))
    messages.append(response.message)

    for call in response.message.tool_calls or []:
        result = run_tool(call.function.name, call.function.arguments or {})
        print(f"🔧 {call.function.name}() → {result}")
        messages.append({"role": "tool", "tool_name": call.function.name, "content": result})

    final = ollama.chat(model="qwen3:4b-instruct", messages=messages)
    print("\n" + final.message.content)
