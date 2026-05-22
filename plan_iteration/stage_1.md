# 阶段 1：通用基座大规模预训练
    定位：打底通用语言、百科、常识、基础文理知识
    数据总量：15M Tokens
    数据集： 中文（60%）：https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
            英文（40%）：https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
## 数据集预处理
1. 下载 Fineweb 中文数据、RedPajama 英文数据

2. 原始字段抽取
   Fineweb 只取 text 字段。 原始字段：['text', 'source', 'source_from_meta']
   RedPajama 只取raw_content 字段并且映射为text。原始字段：['raw_content', 'doc_id', 'meta', 'quality_signals']

3. 基础清洗
   去 HTML
   去 URL
   去异常符号
   去空文本
   去过短文本
   去超长异常文本

4. 质量过滤
   中文数据：中文字符比例过低的丢弃
   英文数据：英文字符比例过低的丢弃
   标点/数字/乱码比例异常的丢弃

5. 去重
   近似去重

6. 文档级划分
   先分别将chinese和english数据集划分为 train / val / test （8:1:1）后再两两合并 train / val / test (8:1:1)

7. tokenize
   每篇文档 encode 成 token ids
   每篇文档末尾加 eos_token_id

8. 拼接
   train 文档拼成 train_token_stream
   val 文档拼成 val_token_stream
   test 文档拼成 test_token_stream

9. 转为.bin格式
   读取每个 split 的 jsonl，把 input_ids 展平写入连续的 bin 文件，保持原有切分结构不变。

10. 训练
   model(x)
   causal LM loss


## 架构层面
CSA 压缩稀疏注意力 已完成  
HCA 重度压缩稠密注意力  已完成   
滑动窗口注意力 (SWA) 已完成 
Attention Sink    已完成  
流形约束超连接 (mHC)    已完成 
DeepGEMM 融合算子    ❌    使用标准PyTorch，无自定义CUDA核（不需要实现）
FP8 × FP4 混合精度   ❌    当前使用bf16（不需要实现）
Muon 二阶优化器   已完成 


