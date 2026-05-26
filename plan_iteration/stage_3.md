# 阶段 3：分领域独立专家训练（SFT + GRPO）完整实施文档

## 1. 阶段定位

阶段 3 的目标是对已经完成阶段 1 通用预训练和阶段 2 扩窗续训的基础模型进行**分领域专家训练**。

本阶段不追求整体模型能力的统一对齐，而是让不同 MoE 专家分别在各自领域内获得更强的专业能力。

本项目阶段 3 采用 3 个领域专家：

| 领域   |     专家编号 | 目标能力                    |
| ---- | -------: | ----------------------- |
| 数学推理 | expert 0 | 解题、推理、公式推导、最终答案生成       |
| 代码生成 | expert 1 | 代码生成、代码解释、简单调试、测试用例理解   |

阶段 3 的核心原则：

```text
固定 V4 主干，只训练对应领域的专家分支。
```

因此，本阶段不是继续训练整个模型，也不是普通的全参数指令微调，而是：

```text
stage2 base model
→ 冻结主干参数
→ 强制路由到指定领域 expert
→ 只训练该 expert 的 FFN 参数
→ 每个领域独立得到一个专家 checkpoint
```

---

## 2. 阶段 3 总体目标

阶段 3 需要完成以下工作：

1. 准备数学、代码 2 个领域的 SFT 数据。
2. 准备数学、代码 2 个领域的 GRPO prompt pool。
3. 明确区分单轮和多轮数据。
4. 从阶段 2 checkpoint 出发，分别训练 2 个领域专家。
5. 每个专家先做 SFT，再做 GRPO。
6. 训练结束后产出 2 个专家 checkpoint，供阶段 4 OPD 蒸馏合并使用。

---

## 4. 数据集来源

### 4.1 数学专家

SFT 数据：

| 数据集                    |  占比 | Token |
| ---------------------- | --: | ----: |
| AI-MO/NuminaMath-CoT   | 100% | 1.8M |
样本展示：
source（不需要）  problem（需要）   solution（需要）    messages（不需要）

GRPO 数据：

| 数据集          |   占比 | 用途           |
| ------------ | ---: | ------------ |
| openai/gsm8k | 100% | 最终答案可验证的数学推理 |

样本展示：
question（需要）    answer（需要）


### 4.2 代码专家

SFT 数据：

| 数据集                                     |  占比 | Token |
| --------------------------------------- | --: | ----: |
| ise-uiuc/Magicoder-OSS-Instruct-75K     | 60% | 1.08M |
| m-a-p/CodeFeedback-Filtered-Instruction | 40% | 0.72M |

样本展示：
ise-uiuc/Magicoder-OSS-Instruct-75K：
lang（不需要）  raw_index（不需要）   index（不需要）   seed（不需要）    openai_fingerprint（不需要）    problem（需要）   solution（需要）

m-a-p/CodeFeedback-Filtered-Instruction：
query（需要）     answer（需要）      resource（不需要）      lang（不需要）

GRPO 数据：

| 数据集             |   占比 | 用途            |
| --------------- | ---: | ------------- |
| codeparrot/apps | 100% | 代码生成 + 单元测试奖励 |

样本示例：
'problem_id'（不需要）, 'question'（需要）, 'solutions'（需要）, 'input_output'（不需要）, 'difficulty'（不需要）, 'url'（不需要）, 'starter_code'

---

## 7. 阶段 3 数据统一格式

### 7.1 SFT 原始数据格式

SFT 数据统一保存为 JSONL，每行一个样本。

推荐字段：

```json
{
  "id": "sample_id",
  "domain": "math",
  "expert_id": 0,
  "sample_type": "task",
  "source": "UltraData-Math",
  "messages": [
    {
      "role": "system",
      "content": "你是一个数学推理专家。请给出清晰、严谨的解题过程，并在最后给出答案。"
    },
    {
      "role": "user",
      "content": "若 x + y = 10，xy = 21，求 x^2 + y^2。"
    },
    {
      "role": "assistant",
      "content": "由公式 x^2 + y^2 = (x + y)^2 - 2xy，可得 x^2 + y^2 = 100 - 42 = 58。因此答案是 58。"
    }
  ],
  "token_count": 87
}
```

### 7.2 SFT 字段解释

| 字段            | 说明                                                               |
| ------------- | ---------------------------------------------------------------- |
| id            | 样本唯一 ID                                                          |
| stage         | 固定为 stage3_sft                                                   |
| domain        | math / code / science                                            |
| expert_id     | math=0，code=1，science=2                                          |
| dialogue_type | single_turn / multi_turn                                         |
| sample_type   | task / multi_turn_task / domain_identity / boundary / correction |
| source        | 原始数据集名称                                                          |
| messages      | ChatML 风格对话数据                                                    |
| token_count   | tokenizer 后的 token 数                                             |

---

## 8. SFT 数据示例

```json
{
  "id": "math_sft_single_000001",
  "domain": "math",
  "expert_id": 0,
  "sample_type": "task",
  "source": "UltraData-Math",
  "messages": [
    {
      "role": "system",
      "content": "你是一个数学推理专家。请给出清晰的解题过程，并在最后给出答案。"
    },
    {
      "role": "user",
      "content": "若 x + y = 10，xy = 21，求 x^2 + y^2。"
    },
    {
      "role": "assistant",
      "content": "由公式 x^2 + y^2 = (x + y)^2 - 2xy，可得 x^2 + y^2 = 100 - 42 = 58。因此答案是 58。"
    }
  ],
  "token_count": 87
}
```
---

## 11. SFT tokenized 后的数据格式

原始 JSONL 不能直接用于模型训练，需要经过 tokenizer 和 chat template 处理。

tokenized 后建议格式：

```python
{
    "input_ids": [...],
    "attention_mask": [...],
    "labels": [-100, -100, ..., 314, 271, 58, 2],
    "expert_id": 0,
    "domain_id": 0,
    "sample_type_id": 0
}
```

关键要求：

```text
system 和 user 部分：labels = -100
assistant 部分：labels = 正常 token id
```

也就是说，只让模型学习 assistant 的回答，不让模型学习复读 system 和 user prompt。

---

## 12. SFT 数据处理流程

每个领域的数据处理流程：

```text
原始数据集
→ 字段清洗
→ 转成 messages 格式
→ 标记 domain / expert_id / dialogue_type / sample_type
→ 去重
→ 过滤过短和过长样本
→ tokenization
→ 统计 token_count
→ 按目标 token budget 抽样
→ 切分 train / valid
→ 保存 JSONL 和 tokenized dataset
```

### 12.1 推荐过滤规则

通用过滤：

```text
去掉空 prompt
去掉空 answer
去掉乱码比例过高的样本
去掉重复样本
去掉长度超过 max_seq_len 的样本，或进行截断
去掉明显包含广告、HTML 垃圾、异常字符的样本
```

数学过滤：

```text
保留有明确题目和解答的样本
优先保留包含最终答案的样本
去掉只有答案没有过程的低质量样本，除非作为少量 short-answer 数据
```

代码过滤：

```text
保留 instruction + code answer
去掉代码块无法解析且内容明显残缺的样本
去掉过长代码文件型样本
尽量保留函数级、算法题级、调试级样本
```

---

## 13. GRPO 数据格式

GRPO 数据不使用 assistant 标准答案作为 labels。
---

## 14. GRPO 数据示例
### 数学专家 (expert_id: 0)
```json
{
  "id": "gsm8k_000001",
  "domain": "math",
  "expert_id": 0,
  "source": "openai/gsm8k",
  "prompt": [
    {
      "role": "system",
      "content": "你是数学推理专家。请逐步推理，并在最后用“答案：”给出最终答案。"
    },
    {
      "role": "user",
      "content": "Janet has 3 apples and buys 5 more. How many apples does she have?"
    }
  ],
  "reward_type": "math_exact",
  "reward_meta": {
    "answer": "8"
  }
}

打分依据：
答案正确性
```

### 代码专家 (expert_id: 1)

```json
{
  "id": "apps_000001",
  "domain": "code",
  "expert_id": 1,
  "source": "codeparrot/apps",
  "prompt": [
    {
      "role": "system",
      "content": "你是代码生成专家。请编写解决以下问题的Python代码，只输出代码，不需要解释。"
    },
    {
      "role": "user",
      "content": "编写一个函数计算两个数的最大公约数。"
    }
  ],
  "reward_type": "unit_test",
  "reward_meta": {
    "test_cases": [
      {
        "input": "gcd(12, 8)",
        "expected_output": "4"
      },
      {
        "input": "gcd(17, 13)",
        "expected_output": "1"
      },
      {
        "input": "gcd(100, 25)",
        "expected_output": "25"
      }
    ],
    "language": "python",
    "time_limit": 2.0
  }
}

示例：
# 模型生成的代码（字符串）
generated_code = '''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
'''

# 数据中的测试用例
test_cases = [
    {"input": "gcd(12, 8)", "expected_output": "4"},
    {"input": "gcd(17, 13)", "expected_output": "1"}
]

# 奖励函数执行流程
def code_reward_single(generated_code, test_cases):
    score = 0
    
    # 1. 执行代码字符串，定义函数
    local_env = {}
    try:
        exec(generated_code, {}, local_env)
    except SyntaxError:
        return 0.0  # 语法错误，0分
    
    # 2. 执行测试
    for test in test_cases:
        try:
            # eval("gcd(12, 8)") → 调用函数，返回 4
            actual_output = eval(test["input"], {}, local_env)
            
            if str(actual_output) == test["expected_output"]:
                score += 1 / len(test_cases)  # 通过，加分
        except Exception as e:
            pass  # 执行出错，不得分
    
    return score

# 运行
reward = code_reward_single(generated_code, test_cases)
print(reward)  # 输出: 1.0 (全部通过)

打分依据：
答案正确性
耗时
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `reward_type` | 奖励函数类型，决定使用哪个 reward_fn |
| `reward_meta` | 奖励函数打分所需的验证信息，各字段含义因领域而异 |

各 `reward_meta` 字段详细说明：

- **数学**: `answer` - 标准答案，用于 exact match 验证
- **代码**: `test_cases` - 测试用例列表；`language` - 编程语言；`time_limit` - 执行超时限制

---

## 15. 是否需要 LoRA

阶段 3 正式训练不建议默认使用 LoRA。

原因：

1. 本项目模型规模约 5M 参数，直接训练专家 FFN 成本很低。
2. 阶段 3 的目标是训练 MoE expert 本体，不是给模型外挂 adapter。
3. 直接训练 expert 权重更利于阶段 4 的 OPD 蒸馏合并。
4. LoRA 会增加权重合并和后续专家路径蒸馏的复杂度。

推荐方案：

```text
不用 LoRA。
冻结主干。
冻结 gate。
冻结 shared expert。
冻结其他 routed experts。
只训练当前领域 expert 的 FFN 权重。
```

## 16. 阶段 3 模型训练策略

### 16.1 专家分配

假设模型有 2 个 routed experts：

| expert_id | 用途        |
| --------: | --------- |
|         0 | 数学专家      |
|         1 | 代码专家      |

shared expert 保持冻结。

### 16.2 训练时强制路由

阶段 3 训练时需要强制所有 token 进入当前领域专家。

原因：

1. 避免 gate 尚未学会领域路由时训练不稳定。
2. 保证对应领域样本只更新对应 expert。
3. 保持其他专家不被污染。
4. 阶段 4 再通过 OPD 学习统一路由和路径融合。

建议在 MoE forward 中加入：

```python
def forward(self, x, force_expert_id: int | None = None):
    if force_expert_id is not None:
        expert_out = self.experts[force_expert_id](x)
        shared_out = self.shared_experts(x)
        return expert_out + shared_out

    # 正常 hash route 或 learned gate route
    ...
```

如果 shared expert 已冻结，可以正常参与前向，但不更新参数。

训练调用：

```python
# 数学
model(input_ids, labels=labels, force_expert_id=0)

# 代码
model(input_ids, labels=labels, force_expert_id=1)
```

---

## 17. 参数冻结策略

阶段 3 每次只训练一个专家。

示例：

```python
def freeze_all(model):
    for name, p in model.named_parameters():
        p.requires_grad = False


def unfreeze_domain_expert(model, expert_id: int):
    for layer in model.layers:
        expert = layer.moe.experts[expert_id]
        for p in expert.parameters():
            p.requires_grad = True
```

训练数学专家：

```python
model = load_checkpoint("checkpoints/stage2/base.pt")
freeze_all(model)
unfreeze_domain_expert(model, expert_id=0)
```

训练代码专家：

```python
model = load_checkpoint("checkpoints/stage2/base.pt")
freeze_all(model)
unfreeze_domain_expert(model, expert_id=1)
```

注意：每个领域都应该从同一个 stage2 base checkpoint 开始，而不是串行训练。

## 18. SFT 训练流程

### 18.1 单领域 SFT 流程

以数学专家为例：

```text
加载 stage2 base checkpoint
→ 冻结全部参数
→ 解冻 expert 0
→ 加载 math SFT 数据
→ force_expert_id = 0
→ 计算 assistant labels 的 causal LM loss
→ 更新 expert 0
→ 保存 math_sft.pt
```

代码专家：

```text
加载 stage2 base checkpoint
→ 冻结全部参数
→ 解冻 expert 1
→ 加载 code SFT 数据
→ force_expert_id = 1
→ 更新 expert 1
→ 保存 code_sft.pt
```

---

## 19. GRPO 训练流程

### 19.1 GRPO 的作用

SFT 让专家学会领域任务的基本回答形式。

GRPO 在 SFT 基础上进一步强化可验证能力：

| 领域 | GRPO 强化目标    |
| -- | ------------ |
| 数学 | 最终答案正确率      |
| 代码 | 单元测试通过率      |


### 19.2 GRPO 基本思想

对同一个 prompt，一次采样多个回答：

```text
y1, y2, y3, y4
```

用 reward function 打分：

```text
r1, r2, r3, r4
```

在组内做归一化：

```text
A_i = (r_i - mean(r)) / (std(r) + eps)
```

然后让高分回答的概率上升，低分回答的概率下降，同时用 KL 约束模型不要偏离 SFT reference model 太远。

### 19.3 GRPO 初始化

每个领域 GRPO 都从自己的 SFT checkpoint 开始：

```text
math_sft.pt → math_grpo.pt
code_sft.pt → code_grpo.pt
```

其中：

```text
policy_model = 当前正在训练的模型
reference_model = 该领域 SFT 后的冻结模型
```

---

## 20. GRPO 奖励函数设计

### 20.1 数学 reward

奖励目标：最终答案正确。

建议规则：

```text
最终答案正确：+1.0
没有最终答案：-0.2
输出过长：-0.05 ~ -0.2
```

示例：

```python
def math_reward(completions, answer, **kwargs):
    rewards = []
    for text, gold in zip(completions, answer):
        pred = extract_final_answer(text)

        score = 0.0
        if pred is not None and normalize_math(pred) == normalize_math(gold):
            score += 1.0

        if "答案" in text or "\\boxed" in text:
            score += 0.1

        if pred is None:
            score -= 0.2

        if len(text) > 1200:
            score -= 0.1

        rewards.append(score)
    return rewards
```
### 20.2 代码 reward

奖励目标：生成代码能通过测试。

建议规则：

```text
通过部分测试：0 ~ +1.0
全部测试通过：+1.0
运行错误：-0.2
超时：-0.3
```

示例：

```python
def code_reward(completions, test_cases, **kwargs):
    rewards = []
    for code, tests in zip(completions, test_cases):
        score = 0.0

        if is_valid_python(code):
            score += 0.1

        passed, total = run_unit_tests_safely(code, tests)
        score += passed / max(total, 1)

        if has_timeout_or_runtime_error(code, tests):
            score -= 0.3

        rewards.append(score)
    return rewards
```

---
## 26. 阶段 3 评估与验收

### 26.1 数学专家验收

评估集：

```text
GSM8K validation subset
自建 100 ~ 300 条数学题
```

指标：

```text
exact match
最终答案提取成功率
平均输出长度
格式合规率
```

重点检查：

```text
是否能给出最终答案
是否出现废话式长推理
是否经常算错简单算术
是否能解释步骤
```

### 26.2 代码专家验收

评估集：

```text
APPS validation subset
HumanEval 风格小集合
自建函数题
```

指标：

```text
pass@1
语法错误率
运行错误率
超时率
平均输出长度
```

重点检查：

```text
是否输出完整代码
是否能通过基础测试
是否经常漏掉边界条件
是否输出无关解释
```

指标：

```text
label accuracy
evidence match rate
hallucination rate
unknown handling rate
```

参数配置：
1. 训练数据参数
参数	数学专家	代码专家
train_val_split   90:10   90:10 
num_generations (G)   4     4
2. 生成采样参数
参数	数学专家	代码专家	说明
temperature 0.7 0.8 代码稍高增加多样性
max_new_tokens  256 512 代码通常更长
3. 优化器参数（Muon + AdamW 混合）
参数	值	说明
optimizer_type  FFN  
learning_rate 5e-5  比 SFT 略低，GRPO 更不稳定
weight_decay  0.01
lr_scheduler  cosine
grad_clip_norm  1.0 必须裁剪，GRPO 梯度方差大
adamw_lr (embed/head) 1e-4
4. GRPO 算法参数
参数	值	说明
kl_coef (β_kl)  0.01  KL 约束系数，防止偏离 SFT 太远
clip_eps (ε)  0.2 PPO 裁剪范围 [1-ε, 1+ε]
group_size (G)  4 每组采样回答数
normalize_advantage True  组内归一化必须开
5. 训练控制参数
参数	数学专家	代码专家
max_seq_len 1024  1024