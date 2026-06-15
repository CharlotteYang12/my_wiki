# Wiki 索引

## Agent

| 页面 | 摘要 |
| --- | --- |
| [[agent]] | Agent 技术总览：基本架构、规划、工具调用、记忆、反思、多智能体，以及 OpenClaw / Hermes / Cursor 对比。 |
| [[agent-react]] | **ReAct: Synergizing Reasoning and Acting in Language Models**：把推理和行动放进 `Thought -> Action -> Observation` 循环，并结合 CoT-SC 提升表现。 |
| [[agent-Memory in the Age of AI Agents]] | **Memory in the Age of AI Agents: A Survey**：Agent 记忆综述入口，目前正文待补充。 |

## 强化学习与世界模型

| 论文 | 摘要 |
| --- | --- |
| [[dreamerv4]] | **Training Agents Inside of Scalable World Models**：DreamerV4 的世界模型、离线强化学习、想象训练、PMPO、value head / actor head 和 Shortcut Models。 |
| [[unsupervised_rl_1000layer ]] | **Unsupervised RL with 1000 Layers**：用对比式 goal-conditioned RL 和残差网络把自监督 RL 扩展到 1000 层，展示深度缩放带来的探索、表征和泛化能力。 |
| [[rl-flow_grpo]] | **Flow-GRPO**：把 flow matching 采样改造成可探索的 SDE 过程，并用 GRPO 在线优化图像生成 reward。 |
| [[rl-test_time_gradient_guidance_flow_policies]] | **Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning**：在推理时用 Q 函数梯度引导 flow policy 的 denoising 过程，提高离线 RL 动作质量。 |

## 大模型后训练

| 文章 | 摘要 |
| --- | --- |
| [[sft_rl]] | **SFT, RL, and On-Policy Distillation Through a Distributional Lens**：从分布塑形角度比较 SFT、RL 和 OPD，解释 on-policy 数据为什么能减少遗忘。 |
| [[training-gft]] | **GFT: From Imitation to Reward Fine-Tuning**：把 SFT 解释为退化 policy gradient，并用 group advantage 与动态系数修正兼顾知识注入、训练稳定和后续 RL 探索空间。 |
| [[rl_basic]] | **RL 基础概念**：区分 reward、value、Q 和 advantage，并梳理 REINFORCE、PPO、Actor-Critic、Q-learning、GRPO 与 Q-Guided Flow 的核心差异。 |

## 机器人控制

| 论文 | 摘要 |
| --- | --- |
| [[bfm-zero]] | 机器人运动控制笔记：Unitree G1 的自由度分布、当前观测维度和历史观测维度计算。 |
| [[EmbodiedMidtrain260421]] | **EmbodiedMidtrain**：通过轻量分类器从 VLM 数据中筛选更接近 VLA 分布的数据，用中段训练提升机器人任务表现。 |
| [[Embodied_Pi]] | **Pi 0.5 / Pi 0.6**：整理具身智能模型的两阶段训练、RECAP、优势条件化、AR 与 Diffusion 双路动作学习。 |
| [[EmbodiedOat]] | **OAT: Ordered Action Tokenization**：为自回归机器人策略设计有序、可解码、可前缀生成的动作 token，支持 anytime action generation。 |

## 视觉生成

| 论文 | 摘要 |
| --- | --- |
| [[VisualGenerationFDLossfor]] | **Visual Generation FD Loss**：把 FD 从评价指标改造成后训练 loss，通过大样本统计和小 batch 梯度解耦，让一步生成模型在 FD / FID 上显著提升。 |

## 自动驾驶安全

| 论文 | 摘要 |
| --- | --- |
| [[gssm]] | **GSSM**：用条件概率分布和极值思想把交通间距转化为风险等级，用轻量神经网络预测环境相关安全标准。 |
| [[rl-selfplay]] | **GIGAFLOW / nuPlan self-play**：整理自动驾驶自我博弈、大规模仿真、优势过滤，以及 PDM-Hybrid、Diffusion-ES 等 nuPlan 方法对比。 |

## 其他资料

| 页面 | 摘要 |
| --- | --- |
| [[巨头公司融资]] | 大模型公司与具身智能公司的估值、融资阶段、商业数据、技术路线和主要产品。 |
| [[log]] | Wiki 摄取、整理和维护日志。 |
