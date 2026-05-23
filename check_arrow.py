from datasets import load_dataset

# 直接加载单文件
ds = load_dataset(
    "arrow",
    data_files="datasets/stage_3_datasets/math/sft/3_final/val/data-00000-of-00001.arrow",
    split="train"
)

print(f"样本数: {len(ds)}")
print(f"字段: {ds.column_names}")
for i in range(3):
    print(f"\n--- 样本{i} ---")
    print(ds[i])