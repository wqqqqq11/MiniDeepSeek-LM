import json
from collections import defaultdict
from glob import glob
from transformers import AutoTokenizer

def analyze_token_distribution():
    tokenizer = AutoTokenizer.from_pretrained(
        "jingyaogong/minimind-3",
        trust_remote_code=True
    )
    
    domains = ["math", "code", "science"]
    stats = defaultdict(list)
    
    for domain in domains:
        # 读取 1_convert 的数据（messages 格式）
        filepath = f"datasets/stage_3_datasets/{domain}/sft/1_convert/*.jsonl"
        
        for file in glob(filepath):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    
                    if not messages:
                        continue
                    
                    # 实际 tokenize（和 Stage 2 完全一致）
                    try:
                        full_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        token_ids = tokenizer.encode(
                            full_text,
                            add_special_tokens=False
                        )
                        # 添加 eos（和 Stage 2 一致）
                        token_ids.append(tokenizer.eos_token_id)
                        
                        stats[domain].append(len(token_ids))
                    except Exception as e:
                        print(f"处理失败: {e}")
                        continue
    
    # 输出统计
    for domain, counts in stats.items():
        if not counts:
            print(f"\n{domain}: 无数据")
            continue
        sorted_counts = sorted(counts)
        n = len(counts)
        print(f"\n{domain}:")
        print(f"  样本数: {n}")
        print(f"  平均: {sum(counts)/n:.0f}")
        print(f"  中位数: {sorted_counts[n//2]}")
        print(f"  P90: {sorted_counts[int(n*0.9)]}")
        print(f"  P95: {sorted_counts[int(n*0.95)]}")
        print(f"  P99: {sorted_counts[int(n*0.99)]}")
        print(f"  最大: {max(counts)}")
        print(f"  >512: {sum(1 for c in counts if c > 512)} ({sum(1 for c in counts if c > 512)/n*100:.1f}%)")
        print(f"  >1024: {sum(1 for c in counts if c > 1024)} ({sum(1 for c in counts if c > 1024)/n*100:.1f}%)")
        print(f"  >1536: {sum(1 for c in counts if c > 1536)} ({sum(1 for c in counts if c > 1536)/n*100:.1f}%)")
        print(f"  >2048: {sum(1 for c in counts if c > 2048)} ({sum(1 for c in counts if c > 2048)/n*100:.1f}%)")

analyze_token_distribution()