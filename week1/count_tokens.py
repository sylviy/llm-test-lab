import json
import tiktoken
from transformers import AutoTokenizer

texts = json.load(open("texts.json", encoding="utf-8"))
toks = {
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    "o200k_base": tiktoken.get_encoding("o200k_base"),
    "qwen": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
}

for name, text in texts.items():
    print(f"\n== {name}({len(text)} 个字符) ==")
    for tok_name, tok in toks.items():
        n = len(tok.encode(text))
        print(f"  {tok_name:8} {n:4} 个 token，每个 token 约 {len(text)/n:.2f} 个字符")

text_pre = texts["zh"][:10]
for tok_name, tok in toks.items():
    pieces = [tok.decode([i]) for i in tok.encode(text_pre)]
    print(tok_name, pieces)
