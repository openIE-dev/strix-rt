#!/usr/bin/env python3
"""Writes logs/beam/prompts.tsv: name<TAB>ids_csv  (Harmony chat template for chat prompts, raw text for completions)."""
from tokenizers import Tokenizer
import os
t = Tokenizer.from_file(os.path.expanduser("$MODELS/new/beam/nvfp4/tokenizer.json"))
def chat(u): return f"<|start|>user<|message|>{u}<|end|><|start|>assistant"
P = [
 ("paris_raw", "The capital of France is"),
 ("paris_chat", chat("What is the capital of France?")),
 ("math_chat", chat("What is 17 * 23? Think step by step.")),
 ("code_py", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
 ("code_chat", chat("Write a Rust function that reverses a string slice.")),
 ("prose", "It was a bright cold day in April, and the clocks were striking thirteen. Winston"),
 ("science_chat", chat("Explain why the sky is blue in two sentences.")),
 ("history", "The French Revolution began in 1789 when"),
 ("german", "Die Hauptstadt von Deutschland ist Berlin. Die Stadt liegt"),
 ("sql", "SELECT name, COUNT(*) FROM orders JOIN customers ON"),
 ("logic_chat", chat("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer briefly.")),
 ("json", "{\"name\": \"Alice\", \"age\": 30, \"address\": {\"city\":"),
]
os.makedirs(os.path.expanduser("~/strix-rt/logs/beam"), exist_ok=True)
with open(os.path.expanduser("~/strix-rt/logs/beam/prompts.tsv"), "w") as f:
    for n, p in P: f.write(n + "\t" + ",".join(map(str, t.encode(p, add_special_tokens=False).ids)) + "\n")
print(len(P), "prompts")
