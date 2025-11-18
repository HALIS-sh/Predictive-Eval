# Predictive-Eval

RL 训练过程的预测实验：CSV + Behavior + AlphaRL

## 项目目标：
### 1. **在真实 GRPO 轨迹上，用
	- CSV（Capability Salience Vector 风格的能力轴）
	- Behavior（四种推理习惯频次）
	去刻画模型在 RL 过程中能力的演化，并检验它们是否能预测最终性能。
### 2. **验证 Alpha-RL 本身的预测能力：
	用 Alpha-RL 的 Rank-1 / PLS 方法在同一条 RL 轨迹上做 “早期 step → 终点” 的外推，并与真实终点对比。
### 3. **把两条线合在一起：
在真实 RL 轨迹和 Alpha-RL 预测出来的轨迹上，观察 CSV + Behavior 特征是否也满足类似的规律，从而为“零 / 少量 RL 预测最终性能”提供证据。



## 0. 依赖与目录结构

### 0.1 环境依赖
	- Python ### 3.11+
	- torch, transformers, datasets
	- pandas, pyarrow, numpy, scikit-learn
	- pyyaml, tqdm, matplotlib/seaborn（可选，仅画图用）

一个可用的本地 Qwen 模型，例如：
	- /data/wenhesun/model/Qwen/Qwen### 2.5-### 1.5B-Instruct
	- /data/wenhesun/model/Qwen/Qwen3-8B

一条已经用 VERL / GRPO 训练好的 RL 轨迹，例如：
	- /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params/global_step_xxx

一个 Alpha-RL 仓库，例如：
	- /data/wenhesun/Alpha-RL/Alpha-RL

### 0.2 项目结构

```text
project_root/
├─ configs/
│  ├─ data.yaml                # build_probe / learn_task_vocab 等配置
│  ├─ csv.yaml                 # train_csv_weights 配置
│  └─ behavior.yaml            # behavior_label_and_train 配置
├─ data/
│  ├─ probe/                   # probe.jsonl 等
│  ├─ infer/                   # run_inference 产生的 *.parquet
│  ├─ csv/                     # train_csv_weights 输出的能力轴
│  ├─ behavior/                # 行为特征
│  └─ rl/                      # GRPO/AlphaRL 各种曲线与中间结果
├─ scripts/
│  ├─ build_probe.py
│  ├─ learn_task_vocab.py
│  ├─ run_inference.py
│  ├─ train_csv_weights.py
│  ├─ behavior_label_and_train.py
│  ├─ collect_csv_along_grpo.py
│  ├─ collect_rl_curves.py
│  ├─ plot_rl_vs_csv_curves.py
│  ├─ alpharl_step1_svd_and_rank1.py
│  └─ alpharl_step2_predict_u_and_build_model.py
└─ ...


⸻

## 1. 构建统一 Probe 与 Task Vocabulary

### 1.1 构建 probe：build_probe.py

作用
	- 从本地 math / coding / logic 数据集构建统一的 probe 集合；
	- 统一 prompt 模板、语言和答案输出格式（例如答案要求写在 ### <final_answer> 之后）；
	- 输出标准 schema：

{"id", "task", "dataset", "split", "prompt", "answer", "meta": {...}}

输入
	- 各个原始数据集 JSON / HF 数据集
	- configs/data.yaml 中指定使用哪些数据集、采样多少条

输出
	- data/probe/probe.jsonl

示例命令

python scripts/build_probe.py \
  --config configs/data.yaml


⸻

### 1.2 学习 task vocabulary：learn_task_vocab.py

作用
	- 基于 probe.jsonl 和预定义的 token category（math / code / logic / tool / reasoning），统计：
	- 哪些 token / n-gram 在不同 task 中出现频率高（TF-IDF 风格）；
	- 自动挑选每个 task 的 “代表性 token vocabulary”，后续用于 CSV 权重初始化或特征统计。

输入
	- data/probe/probe.jsonl

输出
	- data/probe/task_vocab.json（示意）

示例命令

python scripts/learn_task_vocab.py \
  --probe_path data/probe/probe.jsonl \
  --output_path data/probe/task_vocab.json


⸻

## 2. 在 probe 上统一推断：NLL & token 标记

### 2.1 统一推断：run_inference.py

作用
	- 对每个待评估模型，在 probe.jsonl 上做 teacher-forcing 推断：
	- 拼接 prompt 和 ### <answer>，计算 token 级 NLL；
	- 用简单的词表规则打标每个 token：
	- math_token / code_token / logic_token / tool_token / reason_token
	- 区分 prompt token vs answer token（is_answer_mask）；
	- 输出一个 Parquet，作为后续 CSV 与 Behavior 的基础。

输入
	- 本地 HF 模型目录（如 /data/wenhesun/model/Qwen/Qwen### 2.5-### 1.5B-Instruct）
	- probe.jsonl

输出
	- data/infer/<model_name>.probe.parquet，字段示例：

['id', 'task', 'dataset', 'split', 'text', 'prompt_len',
 'input_ids', 'tokens', 'nll',
 'is_answer_mask',
 'math_mask', 'code_mask', 'logic_mask', 'tool_mask', 'reason_mask',
 'mean_nll', 'mean_nll_answer_only']

示例命令

python scripts/run_inference.py \
  --model_path /data/wenhesun/model/Qwen/Qwen### 2.5-### 1.5B-Instruct \
  --probe_path data/probe/probe.jsonl \
  --output_path data/infer/Qwen### 2.5-### 1.5B-Instruct.probe.parquet \
  --batch_size 4 \
  --max_length 2048 \
  --answer_tag "###"


⸻

## 3. 训练 CSV 权重 + 抽取行为特征

### 3.1 Capability Salience：train_csv_weights.py

作用
	- 类似 CSV 论文（Capability Salience Vector）的简化版：
	- 读取 probe_infer.parquet 中的 token NLL 和 token category（math / code / …）；
	- 根据 configs/csv.yaml 定义的任务桶（math / coding / logic 等），学习每个任务的 salience 权重和非线性映射；
	- 得到每个 模型 × 任务 的 “CSV 能力分数”。

输入
	- data/infer/*.probe.parquet
	- configs/csv.yaml

输出
	- data/csv/csv_weights.json（每任务的权重与映射参数）
	- data/csv/csv_scores.parquet（形如 [model_name, task, csv_score]）

示例命令

python scripts/train_csv_weights.py \
  --config configs/csv.yaml \
  --infer_dir data/infer \
  --output_dir data/csv


⸻

### 3.2 行为探针：behavior_label_and_train.py

作用
	- 对 probe 上模型的生成文本进行 “四习惯” 行为标注（可以使用：
	- LLM 打标签；
	- 或一个轻量多标签分类器）；
	- 统计每个 模型 × 任务 的习惯频次及分布（是否出现在末段等），形成行为特征向量。

输入
	- probe 的生成 & loss 信息（通常来自 data/infer/*.probe.parquet）
	- LLM / 分类器配置（configs/behavior.yaml）

输出
	- data/behavior/behavior_raw.parquet（按 probe 细粒度记录）
	- data/behavior/behavior_feats.parquet（按模型聚合后的 4–8 维行为特征）

示例命令

python scripts/behavior_label_and_train.py \
  --config configs/behavior.yaml \
  --infer_dir data/infer \
  --output_dir data/behavior


⸻

### 4. 在真实 GRPO 轨迹上观察 CSV + RL 曲线

这一部分只依赖 真实 GRPO 轨迹，不使用 Alpha-RL。

### 4.1 沿 RL 轨迹收集 CSV：collect_csv_along_grpo.py

作用
	- 遍历 RL 轨迹目录中的若干 global_step_xxx 模型；
	- 对每个 step：
	1. 调用 run_inference.py 得到 *.probe.parquet
	2. 利用已经训练好的 csv_weights 计算这个 step 的 CSV 能力分数
	- 整理成 step → CSV score 的轨迹。

输入
	- base 模型（可选，用于归一化）
	- RL 轨迹根目录：/data/wenhesun/checkpoints/grpo_dapo_math_17k/.../global_step_xxx
	- probe.jsonl
	- csv_weights.json

输出
	- data/rl/csv_tracks.parquet，示例字段：

[step, model_path, task, csv_score, model_name, algo, ...]

示例命令

python scripts/collect_csv_along_grpo.py \
  --probe_path data/probe/probe.jsonl \
  --csv_weights_path data/csv/csv_weights.json \
  --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
  --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
  --output_path data/rl/csv_tracks.parquet \
  --start_step 2000 --end_step 17000 --step_stride 2000


⸻

### 4.2 收集真实 RL 曲线：collect_rl_curves.py

作用
	- 使用你已有的 eval pipeline（或 AlphaRL 的 data_eval.py），对每个 global_step_xxx 进行评测；
	- 输出 step → 真实 RL score 的曲线。

输入
	- RL 轨迹根目录
	- eval 脚本（如 Alpha-RL data_eval.py）及其配置（dataset, max_examples 等）

输出
	- data/rl/rl_curves.parquet，字段示例：

[step, rl_score, metric_key, model_path, dataset, ...]

示例命令

python scripts/collect_rl_curves.py \
  --alpharl_root /data/wenhesun/Alpha-RL/Alpha-RL \
  --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
  --output_path data/rl/rl_curves.parquet \
  --metric_key acc \
  --eval_extra_args "--dataset gsm8k --max_examples 500" \
  --start_step 2000 --end_step 17000 --step_stride 2000


⸻

### 4.3 画 RL vs CSV 曲线 + 回归：plot_rl_vs_csv_curves.py

作用
	- 读取 rl_curves.parquet 和 csv_tracks.parquet；
	- 按 step merge；
	- 作图：
	- step → RL score 曲线
	- step → CSV score 曲线（可归一化）；
	- 做一维回归：
	- RL_score ~ CSV_score，输出 R²、Pearson、Spearman 等。

输入
	- data/rl/rl_curves.parquet
	- data/rl/csv_tracks.parquet

输出
	- data/rl/rl_vs_csv.png 等图像
	- 终端打印 R² / 相关系数等

示例命令

python scripts/plot_rl_vs_csv_curves.py \
  --rl_curves data/rl/rl_curves.parquet \
  --csv_tracks data/rl/csv_tracks.parquet \
  --output_dir data/rl/plots \
  --task math

通过这一步，你可以回答：“在真实 GRPO 轨迹里，CSV 能力轴是否单调/线性地跟 RL 性能相关？”

⸻

## 5. 验证 Alpha-RL 自身效果：SVD + Rank-1 + PLS

下面两步是对 Alpha-RL 论文结果的复现 / 验证，完全基于 alphaRL 仓库中的工具脚本。

### 5.1 Step1：SVD + Rank-1 重建：alpharl_step1_svd_and_rank### 1.py

作用
	1. 调用 Alpha-RL 的 svd.save_svd_components：
用 base 模型与每个 global_step_xxx 做差，对各层 weight 做 SVD，并把 U,S,Vt 存在：

<output_root>/svd_components/global_step_xxx/svd_components.pt


	2. 调用 Alpha-RL 的 upd_rank.reconstruct_and_save_rank1：
在 base 模型上，用 top-1 SVD 方向重建 Rank-1 模型：

<output_root>/svd_components/global_step_xxx/rank_1/


	3. 使用 Alpha-RL 的 data_eval.py 对：
	- 真实 RL ckpt (ckpt_root/global_step_xxx)
	- Rank-1 模型 (svd_components/global_step_xxx/rank_1)
分别评测，保存 metrics json，并汇总成一条 Parquet 曲线。

输入
	- Alpha-RL 根目录：/data/wenhesun/Alpha-RL/Alpha-RL
	- base 模型路径：/data/wenhesun/model/Qwen/Qwen3-8B
	- RL 轨迹根目录
	- eval 参数（dataset / max_examples / metric_key）

输出
	- output_root/svd_components/...（SVD 中间结果 + rank_1 模型）
	- output_root/metrics_rl/*.json
	- output_root/metrics_rank1/*.json
	- output_root/alpharl_rank1_eval.parquet

示例命令

python scripts/alpharl_step1_svd_and_rank### 1.py \
  --alpharl_root /data/wenhesun/Alpha-RL/Alpha-RL \
  --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
  --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
  --output_root /data/wenhesun/alpharl_out/grpo_math_qwen3_8b \
  --start_step 2000 --end_step 17000 --step_stride 2000 \
  --eval_extra_args "--dataset gsm8k --max_examples 500" \
  --metric_key acc \
  --experiment_id grpo_math_qwen3_8b \
  --base_model_name Qwen3-8B \
  --algo GRPO \
  --task math \
  --base_score 0.30

用 alpharl_rank1_eval.parquet 可以画出：真实 RL 曲线 vs Rank-1 曲线，并计算 “rank1 恢复的增益比例”。

⸻

### 5.2 Step2：预测终点方向 + 构建预测模型：alpharl_step2_predict_u_and_build_model.py

作用
	1. 从上一步的 SVD 输出中抽取每个 step 的第一向量 first_u_vectors.pt （调用 Alpha-RL 的 first_vector.process_all_steps）。
	2. 读取 alpharl_rank1_eval.parquet 中的 step → rl_score，把 [pls_start_step, pls_end_step] 的点作为 PLS 训练数据。
	3. 调用 Alpha-RL 的 pred.predict_all_keys_pls：
	- 在 early steps 的 (u_step, rl_score_step) 上做 PLS；
	- 以 target_score（例如真实终点得分或目标得分）反推出目标方向 u_pred；
	- 保存到 <output_root>/pred/predicted_u.pt。
	4. 调用 Alpha-RL 的 predupd.reconstruct_and_save_rank1_with_pred_u：
	- 使用 base、SVD 组件、u_pred 构建一个 “预测终点 rank-1 模型” rank1_predu。
	5. 用 data_eval.py 对该预测模型评测，并与真实最终 ckpt 对比；
	6. 输出一个汇总 Parquet。

输入
	- Alpha-RL 根目录
	- base 模型 / ckpt 根目录
	- Step1 输出的 alpharl_rank1_eval.parquet
	- PLS 拟合窗口 step（pls_start_step, pls_end_step）
	- target_score（可以选：
	- auto_final：自动使用真实最终 step 的 rl_score；
	- max：使用 RL 曲线最大值；
	- 或一个具体的浮点数）

输出
	- output_root/pred/predicted_u*.pt
	- svd_components/.../rank1_predu（预测终点模型目录）
	- output_root/metrics_predicted/metrics_predicted_step*.json
	- output_root/alpharl_predicted_end.parquet

示例命令

python scripts/alpharl_step2_predict_u_and_build_model.py \
  --alpharl_root /data/wenhesun/Alpha-RL/Alpha-RL \
  --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
  --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
  --output_root /data/wenhesun/alpharl_out/grpo_math_qwen3_8b \
  --summary_parquet /data/wenhesun/alpharl_out/grpo_math_qwen3_8b/alpharl_rank1_eval.parquet \
  --pls_start_step 2000 \
  --pls_end_step 8000 \
  --target_score auto_final \
  --metric_key acc \
  --eval_extra_args "--dataset gsm8k --max_examples 500" \
  --experiment_id grpo_math_qwen3_8b \
  --base_model_name Qwen3-8B \
  --algo GRPO \
  --task math

跑完之后，alpharl_predicted_end.parquet 会告诉你：
真实终点得分 vs Alpha-RL 预测终点得分，从而验证 Alpha-RL 在你这条 GRPO 轨迹上的可用性。

⸻

### 6. 把 CSV + Behavior 接到 Alpha-RL 的预测模型上

在完成第 5 步后，你会额外得到一个或多个 Alpha-RL 预测出来的 “终点模型”（目录 rank1_predu）。

接下来可以重复第 2–3–4 步中的若干流程，在这些模型上抽取 CSV + Behavior 特征：
	1. 对 rank1_predu 模型跑一次 run_inference.py；
	2. 用 train_csv_weights.py 中的已有权重，对它算 CSV 任务能力；
	3. 用 behavior_label_and_train.py，对它的输出做行为特征统计；
	4. 把这些特征与：
	- base 模型
	- 真实最终 GRPO 模型
进行对比。

可以重点看：
	- Alpha-RL 预测模型在 CSV 能力轴上，是否更接近真实最终模型；
	- CSV 能力 vs RL 性能 之间的回归/相关性，在 Alpha-RL 预测点上是否仍然成立；
	- 行为特征（例如 verification / backtracking 的频次）是否也能被 Alpha-RL 预测方向所“恢复”。

⸻

### 7. 实验 Checklist
	1. 单条 GRPO 轨迹的基础验证
	- 使用 collect_rl_curves.py & collect_csv_along_grpo.py + plot_rl_vs_csv_curves.py：
	- 查看 CSV_score vs RL_score 的相关性与 R²；
	- 尝试不同 task（math / coding 等）和不同 probe 设计，比较哪种最稳定。
	2. Alpha-RL Rank-1 能力验证
	- alpharl_step1_svd_and_rank### 1.py：
	- 比较真实 RL 曲线 vs Rank-1 曲线；
	- 统计 “恢复的增益比例”。
	3. Alpha-RL 终点预测能力验证
	- alpharl_step2_predict_u_and_build_model.py：
	- 比较真实终点 vs 预测终点得分；
	- 在不同 PLS 窗口（pls_start_step, pls_end_step）下做敏感性分析；
	- 对比 target 选择方式（auto_final vs max vs 手动设定）。
	4. 在 Alpha-RL 预测模型上检验 CSV + Behavior 的规律
	- 对 rank1_predu 模型跑 CSV & Behavior；
	- 对比：
	- base vs 真终点 vs AlphaRL 预测终点；
	- 分析：
“AlphaRL 预测出的方向是否在 CSV / Behavior 空间中也走对了方向？”
