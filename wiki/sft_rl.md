> SFT、RL 和 OPD 都是在重塑模型的输出分布；真正决定遗忘与泛化差异的，不只是监督信号强弱，而是训练数据是否来自当前模型自己的分布。

# 背景

来源：[SFT, RL, and On-Policy Distillation Through a Distributional Lens](https://nrehiew.github.io/blog/sft_rl_opd/)

这篇文章用“分布塑形”理解后训练：语言模型本质上是序列分布，SFT、RL 和 OPD 的差别在于它们如何定义目标分布，以及这个目标分布离当前模型有多远。

# 方法

## 三种后训练方法

| 方法 | 数据来源 | 优化目标 | 分布效果 | 主要风险 |
| --- | --- | --- | --- | --- |
| SFT | 固定外部数据集 | token 级交叉熵 / forward KL | 直接拉向外部分布 | 数据分布离原模型太远时容易遗忘 |
| RL | 当前策略采样 | 奖励 / advantage | 沿当前模型访问区域中的高奖励方向移动 | 奖励稀疏、样本成本高 |
| OPD | 学生 on-policy 采样 | 教师 logit / reverse KL 匹配 | 在学生自己访问的状态上吸收教师能力 | 教师信号可能有偏，style token 可能被过度学习 |

SFT 会对外部数据集中的 token 施加更均匀的梯度压力，容易同时推动任务相关能力和无关风格；RL 则主要更新当前策略实际采样到、且奖励更高的区域。

![](assets/sft_rl_sft_distribution.svg)

![](assets/sft_rl_rl_distribution.svg)

## OPD 的关键实验

作者在 Minimal Code Editing 任务上先训练了两个 teacher：一个来自 SFT，一个来自 RL。直觉上，RL teacher 更强、更少遗忘，应该产生更好的 OPD student。

但结果是：

- 两个 OPD student 表现非常接近。
- OPD student 甚至能超过原来的 RL teacher。
- 即使用已经发生遗忘的 SFT teacher，OPD student 也没有继承同等程度的遗忘。

这说明 teacher 不是唯一关键变量；**on-policy 数据来源本身非常重要**。学生是在自己的状态分布上接受指导，而不是被迫模仿教师自己的轨迹分布。

OPSD 的 token 级 KL 分析也提醒：高 KL 不一定代表高任务价值，style token 可能比数学 token 更容易产生强信号，因此 OPD 需要 clipping 这类保护。

![](assets/sft_rl_opsd_token.png)

## 为什么 RL 和 OPD 更不容易遗忘

文章更偏好的解释是：on-policy 数据隐含了一种保守约束。

RL 并不是被拉向任意外部分布，而是在当前策略能采样到的区域里，提高高奖励样本概率。换句话说，RL 倾向于找到“离当前模型最近的能完成任务的策略”。

OPD 也有类似特性。虽然它有 teacher signal，但数据来自 student 自己。teacher 只是给 student 当前访问的 prefix 提建议，因此更新仍然被限制在 student 当前分布附近。

![](assets/sft_rl_on_policy.svg)

## 完整后训练流水线

文章认为开放模型常见路线是：

```text
Pretrain -> SFT -> RL -> OPD
```

- SFT 负责格式对齐和基础指令遵循。
- RL 更适合数学、代码等可验证奖励明确的任务。
- OPD 常用于把多个 expert 能力合并进最终模型。
- 创作、知识问答等 reward 更有偏的任务，可能更适合 distillation / self-distillation。

![](assets/sft_rl_mimo_v2_flash.png)

# 实验结论

- SFT 适合冷启动，但容易因为外部分布过强而压掉原有能力。
- RL 少遗忘的关键不一定是显式 KL，而是 on-policy sampling 带来的“近邻解”偏好。
- OPD 的抗遗忘能力也主要来自 on-policy 数据，而不是 teacher 本身有多强。
- OPD 可能让 student 超过 teacher，因为 teacher 是在 student 自己访问的状态上提供 dense signal。
- 未来更理想的算法需要同时具备：distillation 的密集信号、RL 的低偏差目标、on-policy 的保守分布移动。

OPD 的 reward 提升更突然，同时伴随明显 entropy collapse；这符合 reverse KL / mode-seeking 训练容易收缩分布的直觉。

![](assets/sft_rl_reward_entropy_plots.svg)

## 索引信息

> 类别：文章笔记 / 大模型后训练 / 分布视角  
> 索引标签：#SFT #RL #OPD #后训练 #OnPolicy #灾难性遗忘 #分布塑形 #Distillation
