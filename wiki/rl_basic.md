# RL 基础概念：Reward、Value、Q 与 Advantage

> Reward 是外部给分，Value / Q 是从 reward 学出的未来回报估计，Advantage 则是相对 baseline 的优势；不同 RL 方法的差异，核心在于它们直接使用或学习哪一种信号。

## Reward
`reward` 是环境或规则给的即时分数，回答的是：

```text
这个动作/结果现在得几分？
```

例子：

- 数学题答对：`reward = 1`
- 答错：`reward = 0`
- 人类偏好模型打分：`reward = RM(x, y)`
- 机器人往目标靠近一步：`reward = -distance`

Reward 本身不是模型一定要学的东西，它可以来自环境、规则、reward model、verifier。很多 LLM RLVR / GRPO 直接用最终答案 reward。

## Value: V
`V(s)` 是状态价值，回答的是：

```text
从这个状态开始，未来平均能拿多少总回报？
```

它不关心具体选哪个动作，只评价“这个状态好不好”。

```text
V(s) = E[future return | s]
```

用途：当 baseline，降低 policy gradient 方差。

```text
A(s,a) = return - V(s)
```

直觉：不是看“这条轨迹分数高不高”，而是看“它比这个状态下平均水平高多少”。

## Q
`Q(s,a)` 是动作价值，回答的是：

```text
在状态 s 里做动作 a，未来平均能拿多少总回报？
```

```text
Q(s,a) = E[future return | s, a]
```

它比 `V` 多看了动作，所以可以直接用来选动作：

```text
a = argmax_a Q(s,a)
```

在连续控制里，也可以对动作求梯度：

```text
∇_a Q(s,a)
```

你现在打开的 `Q-Guided Flow` 就是这个思路：先训练 flow policy，再在测试时用 critic 的 `∇_a Q(s,a)` 引导动作往更高价值方向走。

## Advantage
`Advantage` 是相对优势：

```text
A(s,a) = Q(s,a) - V(s)
```

它回答的是：

```text
这个动作比当前状态下的平均动作好多少？
```

如果没有显式学 `Q` / `V`，也可以用采样组近似：

```text
A_i = (R_i - mean(R_group)) / std(R_group)
```

这就是 GRPO / GFT 这类 group advantage 的直觉：不用单独训练 critic，而是让同一个 prompt 下的一组回答互相比较。

## 几类方法
| 方法 | 主要用什么 | 直觉 |
| --- | --- | --- |
| REINFORCE | reward / return | 哪条轨迹得分高，就提高它的概率 |
| PPO | reward + value | 用 value 算 advantage，再稳定更新 policy |
| Actor-Critic | policy + V/Q critic | actor 负责行动，critic 负责评价 |
| Q-learning / DQN | Q | 学每个动作的价值，选 Q 最大的动作 |
| DDPG / TD3 / SAC | Q + policy | 连续动作里用 Q 指导 actor |
| GRPO | group reward advantage | 不学 critic，用同组 reward 均值当 baseline |
| Q-Guided Flow | Q gradient | 不重新训练 actor，测试时用 Q 梯度引导动作 |

最短总结：

```text
reward: 外部给分
V: 这个状态整体好不好
Q: 这个状态下做这个动作好不好
advantage: 这个动作比平均水平好多少
```

LLM 里的 GRPO/GFT 更偏 `reward -> advantage -> policy update`；
机器人控制里的 Q-guidance 更偏 `Q(s,a) -> 动作梯度 -> 改动作`。

## 索引信息

> 类别：基础概念 / 强化学习  
> 索引标签：#强化学习 #Reward #Value #QFunction #Advantage #PolicyGradient #ActorCritic #GRPO