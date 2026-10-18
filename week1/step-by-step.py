import tiktoken
from transformers import AutoTokenizer

enc = tiktoken.get_encoding("cl100k_base")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = enc.encode("路由器收到组播报文")
ids_qwen = qwen.encode("路由器收到组播报文")
print(ids)
print(len(ids))
print(ids_qwen)
print(len(ids_qwen))

