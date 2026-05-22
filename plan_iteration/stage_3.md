# 阶段 3：分领域独立专家训练（SFT + GRPO）完整实施文档

## 1. 阶段定位

阶段 3 的目标是对已经完成阶段 1 通用预训练和阶段 2 扩窗续训的基础模型进行**分领域专家训练**。

本阶段不追求整体模型能力的统一对齐，而是让不同 MoE 专家分别在各自领域内获得更强的专业能力。

本项目阶段 3 采用 3 个领域专家：

| 领域   |     专家编号 | 目标能力                    |
| ---- | -------: | ----------------------- |
| 数学推理 | expert 0 | 解题、推理、公式推导、最终答案生成       |
| 代码生成 | expert 1 | 代码生成、代码解释、简单调试、测试用例理解   |
| 科研学术 | expert 2 | 科研文本理解、论文摘要、证据判断、科学事实核查 |

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

1. 准备数学、代码、科研 3 个领域的 SFT 数据。
2. 准备数学、代码、科研 3 个领域的 GRPO prompt pool。
3. 明确区分单轮和多轮数据。
4. 从阶段 2 checkpoint 出发，分别训练 3 个领域专家。
5. 每个专家先做 SFT，再做 GRPO。
6. 训练结束后产出 3 个专家 checkpoint，供阶段 4 OPD 蒸馏合并使用。

---

## 3. 阶段 3 数据规模设计

阶段 3 总数据量：6M tokens。

每个领域 2M tokens：

| 领域   | 总 Token | SFT Token | GRPO Token |
| ---- | ------: | --------: | ---------: |
| 数学推理 |      2M |      1.8M |       0.2M |
| 代码生成 |      2M |      1.8M |       0.2M |
| 科研学术 |      2M |      1.8M |       0.2M |
| 合计   |      6M |      5.4M |       0.6M |

注意：这里的 GRPO Token 不是指标准答案 token，而是指 GRPO 训练中 prompt、生成结果和优化过程中消耗的近似 token budget。

---

## 4. 数据集来源

### 4.1 数学专家

SFT 数据：

| 数据集                    |  占比 | Token |
| ---------------------- | --: | ----: |
| AI-MO/NuminaMath-CoT   | 100% | 1.8M |
样本展示：
source（不需要）  problem（需要）   solution（需要）    messages（不需要）
synthetic_math  Suppose that $g(x) = 5x - 3$. What is $g^{-1}(g^{-1}(14))$? First, we need to find the inverse function $g^{-1}(x)$. G...   [
{
"content": "Suppose that $g(x) = 5x - 3$. What is $g^{-1}(g^{-1}(14))$?",
"role": "user"
},
{
"content": "First, we need to find the inverse function $g^{...

GRPO 数据：

| 数据集          |   占比 | 用途           |
| ------------ | ---: | ------------ |
| openai/gsm8k | 100% | 最终答案可验证的数学推理 |

### 4.2 代码专家

SFT 数据：

| 数据集                                     |  占比 | Token |
| --------------------------------------- | --: | ----: |
| ise-uiuc/Magicoder-OSS-Instruct-75K     | 60% | 1.08M |
| m-a-p/CodeFeedback-Filtered-Instruction | 40% | 0.72M |

样本展示：
ise-uiuc/Magicoder-OSS-Instruct-75K：
lang（不需要）  raw_index（不需要）   index（不需要）   seed（不需要）    openai_fingerprint（不需要）    problem（需要）   solution（需要）
cpp   101,533   4,626   int n;
cin >> n;
vector<int> a(n + 1), b(n + 1);...    fp_eeff13170a   You are given two arrays, A and B, each of length n. You need to perform a convolution...   ```cpp
#include <iostream>
#include <vector>
using namespace std;

vector<int> convolution(vector<int> a, vector<int> b) {...

m-a-p/CodeFeedback-Filtered-Instruction：
query（需要）     answer（需要）      resource（不需要）      lang（不需要）
Create a nested loop to print every combination of numbers between 0-9,...      Here is an example of a nested loop in Python to print every combination of numbers...      evolinstruct      python
GRPO 数据：

| 数据集             |   占比 | 用途            |
| --------------- | ---: | ------------- |
| codeparrot/apps | 100% | 代码生成 + 单元测试奖励 |

### 4.3 科研专家

SFT 数据：

| 数据集              |  占比 | Token |
| ---------------- | --: | ----: |
| qiaojin/PubMedQA | 100% | 1.8M |

样本展示：
pubid（不需要）   question（需要）    context（不需要）     long_answer（需要）     final_decision（不需要）
25,429,730    Are group 2 innate lymphoid cells ( ILC2s ) increased...    {
"contexts": [
"Chronic rhinosinusitis (CRS) is a heterogeneous disease with an uncertain pathog....     As ILC2s are elevated in patients with CRSwNP, they may drive nasal polyp formation in ...    yes
GRPO 数据：

| 数据集             |   占比 | 用途            |
| --------------- | ---: | ------------- |
| allenai/scifact | 100% | 科学声明验证 + 证据判断 |

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

科研过滤：

```text
保留有上下文、摘要、证据、问题、答案的样本
去掉缺少证据却要求具体结论的样本
保留不确定性表达，例如“根据给定内容无法判断”
```

---

## 13. GRPO 数据格式

GRPO 数据不使用 assistant 标准答案作为 labels。

GRPO 数据需要的是：

```text
prompt
+ 可验证答案 / 测试用例 / 标签 / 证据
+ reward_type
```

统一格式：

```json
{
  "id": "sample_id",
  "domain": "math",
  "expert_id": 0,
  "prompt": [
    {
      "role": "system",
      "content": "你是数学推理专家。请逐步推理，并在最后给出答案。"
    },
    {
      "role": "user",
      "content": "..."
    }
  ],
  "reward_type": "math_exact",
  "answer": "..."
}
```

---

## 14. GRPO 数据示例
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
  "answer": "8",
  "reward_type": "math_exact"
}
```
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

假设模型有 3 个 routed experts：

| expert_id | 用途        |
| --------: | --------- |
|         0 | 数学专家      |
|         1 | 代码专家      |
|         2 | 科研专家      |

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

# 科研
model(input_ids, labels=labels, force_expert_id=2)
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

训练科研专家：

```python
model = load_checkpoint("checkpoints/stage2/base.pt")
freeze_all(model)
unfreeze_domain_expert(model, expert_id=2)
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

科研专家：

```text
加载 stage2 base checkpoint
→ 冻结全部参数
→ 解冻 expert 2
→ 加载 science SFT 数据
→ force_expert_id = 2
→ 更新 expert 2
→ 保存 science_sft.pt
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
| 科研 | 标签判断和证据使用正确率 |

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
science_sft.pt → science_grpo.pt
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
格式正确：+0.1
没有最终答案：-0.2
明显胡乱输出：-0.5
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

注意：不要奖励“推理过程很长”。小模型容易学会冗长但错误的 CoT。

### 20.2 代码 reward

奖励目标：生成代码能通过测试。

建议规则：

```text
代码可解析：+0.1
通过部分测试：0 ~ +1.0
全部测试通过：+1.0
运行错误：-0.2
超时：-0.3
输出非代码：-0.2
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

### 20.3 科研 reward

奖励目标：科学声明判断正确，证据使用合理。

建议规则：

```text
标签正确：+0.7
证据选择正确：+0.2
理由简洁且不矛盾：+0.1
编造证据：-0.5
没有给出标签：-0.2
```

示例：

```python
def science_reward(completions, label, evidence_ids=None, **kwargs):
    rewards = []
    for text, gold_label in zip(completions, label):
        pred_label = extract_scifact_label(text)

        score = 0.0
        if pred_label == gold_label:
            score += 0.7

        if contains_reasoning_cue(text):
            score += 0.1

        if hallucinated_citation(text):
            score -= 0.5

        rewards.append(score)
    return rewards
```

科研 reward 不建议使用 BLEU / ROUGE 作为主奖励。

更推荐：

```text
标签是否正确
证据是否匹配
是否编造证据
是否承认信息不足
```

---

## 21. GRPO 训练伪代码

```python
for batch in grpo_loader:
    prompts = batch["prompt"]
    expert_id = batch["expert_id"]

    # 1. 对每个 prompt 采样 G 个回答
    with torch.no_grad():
        completions, old_logprobs = rollout(
            model=policy_model,
            prompts=prompts,
            expert_id=expert_id,
            num_generations=4,
            temperature=0.7,
            max_new_tokens=256,
        )

    # 2. reward function 打分
    rewards = reward_fn(
        completions=completions,
        **batch["reward_meta"]
    )

    # 3. 组内归一化 advantage
    advantages = group_normalize(rewards, group_size=4)

    # 4. 当前 policy 重新计算 logprob
    new_logprobs = compute_logprobs(
        model=policy_model,
        prompts=prompts,
        completions=completions,
        expert_id=expert_id,
    )

    # 5. reference model 计算 logprob
    with torch.no_grad():
        ref_logprobs = compute_logprobs(
            model=reference_sft_model,
            prompts=prompts,
            completions=completions,
            expert_id=expert_id,
        )

    # 6. PPO-style clipped objective
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)

    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # 7. KL 约束
    kl_loss = (new_logprobs - ref_logprobs).mean()

    loss = policy_loss + beta_kl * kl_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
```

---

## 23. 阶段 3 训练顺序

完整训练顺序如下：

```text
Step 1：加载 stage2 base checkpoint

Step 2：建立专家映射
  math    -> expert 0
  code    -> expert 1
  science -> expert 2

Step 3：处理 SFT 数据
  原始数据集
  -> messages 格式
  -> 去重
  -> 长度过滤
  -> tokenization
  -> 按 token budget 抽样

Step 4：数学 SFT
  base_stage2 -> freeze all -> unfreeze expert 0
  force route expert 0
  train 1.8M tokens
  save math_sft.pt

Step 5：代码 SFT
  base_stage2 -> freeze all -> unfreeze expert 1
  force route expert 1
  train 1.8M tokens
  save code_sft.pt

Step 6：科研 SFT
  base_stage2 -> freeze all -> unfreeze expert 2
  force route expert 2
  train 1.8M tokens
  save science_sft.pt

Step 7：处理 GRPO prompt pool
  只保留 prompt + answer/test/evidence
  不保留 assistant labels
  确保和 SFT 样本不重叠

Step 8：数学 GRPO
  policy = math_sft.pt
  reference = math_sft.pt frozen
  reward = math_exact
  save math_grpo.pt

Step 9：代码 GRPO
  policy = code_sft.pt
  reference = code_sft.pt frozen
  reward = unit_test
  save code_grpo.pt

Step 10：科研 GRPO
  policy = science_sft.pt
  reference = science_sft.pt frozen
  reward = scientific_claim_verification
  save science_grpo.pt
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

### 26.3 科研专家验收

评估集：

```text
SciFact validation subset
PubMedQA validation subset
自建 claim/evidence 判断数据
```

指标：

```text
label accuracy
evidence match rate
hallucination rate
unknown handling rate
```
