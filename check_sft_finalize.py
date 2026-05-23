from datasets import load_from_disk

# 加载数据
train = load_from_disk("datasets/stage_3_datasets/math/sft/3_final/train")

# 获取第一个样本
sample = train[0]

print("=" * 60)
print(f"样本 ID: {sample['id']}")
print(f"Domain: {sample['domain']}")
print(f"Expert ID: {sample['expert_id']}")
print(f"Length: {sample['length']}")
print("=" * 60)

input_ids = sample['input_ids']
labels = sample['labels']
attention_mask = sample['attention_mask']

# 找出所有非 -100 的 labels 位置
learn_positions = [i for i, l in enumerate(labels) if l != -100]
print(f"\nLabels 中非 -100 的位置: {len(learn_positions)} 个")
print(f"位置范围: {learn_positions[0] if learn_positions else 'None'} ~ {learn_positions[-1] if learn_positions else 'None'}")

# 查看实际长度范围内的内容
length = sample['length']
print(f"\n实际有效长度 (length): {length}")
print(f"\n实际有效范围内的 tokens:")
print(f"  input_ids[{length-5}:{length}]: {input_ids[length-5:length]}")
print(f"  labels[{length-5}:{length}]: {labels[length-5:length]}")

# 查找 eos token (2)
eos_positions = [i for i, t in enumerate(input_ids) if t == 2]
print(f"\n所有 eos token (2) 的位置: {eos_positions}")

# 查看最后 20 个有效 token
print(f"\n最后 20 个有效 tokens:")
for i in range(max(0, length-20), length):
    label_str = str(labels[i]) if labels[i] != -100 else "-100"
    marker = " <--" if labels[i] != -100 else ""
    print(f"  [{i:4d}] input_id: {input_ids[i]:6d}, label: {label_str:6s}{marker}")