from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('jingyaogong/minimind-3', trust_remote_code=True)

messages = [
    {"role": "system", "content": "test"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"}
]

# 查看 template 输出
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(repr(text))

# 查看编码结果
ids = tokenizer.encode(text, add_special_tokens=False)
print(ids[:10], "...", ids[-10:])
print(f"是否包含 bos(1): {1 in ids}")
print(f"是否包含 eos(2): {2 in ids}")

# python -c "from transformers import AutoTokenizer;tokenizer = AutoTokenizer.from_pretrained('jingyaogong/minimind-3', trust_remote_code=True);print(f'vocab_size: {len(tokenizer)}');print(f'eos_token_id: {tokenizer.eos_token_id}');print(f'pad_token_id: {tokenizer.pad_token_id}');print(f'special_tokens: {tokenizer.special_tokens_map}');print(f'bos_token_id: {tokenizer.bos_token_id}')"