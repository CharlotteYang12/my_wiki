# Wiki 日志

这个文件按时间记录 wiki 的摄取、整理、维护和结构调整。

## [2026-05-14] 初始化 | 建立 LLM Wiki

- 创建初始 wiki 目录结构。
- 添加维护规则文件。
- 添加 `wiki/index.md` 和 `wiki/log.md`。

## [2026-05-16] 维护 | 整理巨头公司融资页

- 将 `wiki/巨头公司融资.md` 从零散记录整理成 Markdown 表格。
- 更新索引中的相关条目。

## [2026-05-16] 维护 | 校验商业数据

- 检查 `wiki/巨头公司融资.md` 中的估值、融资、营收、订单、量产和 IPO 信息。
- 将页面改写为带来源和置信度说明的资料表。

## [2026-05-16] 维护 | 重构融资表结构

- 移除验证列和修正列。
- 将技术路线和产品信息拆成独立列。

## [2026-05-16] 维护 | 统一估值口径

- 加粗公司估值数字。
- 将人民币估值粗略折算成美元口径。

## [2026-05-16] 维护 | 添加融资阶段

- 为 `wiki/巨头公司融资.md` 增加融资阶段列。

## [2026-05-25] 摄取 | DreamerV4 笔记

- 将 DreamerV4 相关原始笔记整理到 wiki。
- 抽取世界模型、想象训练、PMPO 和 Shortcut Models 等重点概念。
- 更新索引条目。

## [2026-05-25] 维护 | 集中图片资源

- 将 wiki 中的图片集中到 `wiki/assets/`。
- 更新 `wiki/bfm-zero.md`、`wiki/agent-react.md` 和 `wiki/dreamerv4.md` 中的图片引用路径。

## [2026-05-25] 维护 | 添加索引标签并重建目录

- 重新读取当前 wiki 页面。
- 为现有页面添加类别和索引标签。
- 按类别重建 `wiki/index.md`，方便后续浏览和检索。

## [2026-05-25] 维护 | 调整索引信息位置

- 将各 wiki 页面顶部的类别和索引标签移动到页面末尾。
- 统一使用 `## 索引信息` 小节承载页面分类与标签。

## [2026-06-01] 摄取 | EmbodiedMidtrain 与 FD-loss 笔记

- 将 `wiki/EmbodiedMidtrain260421.md` 纳入索引，归入机器人控制 / VLA 方向。
- 将 `wiki/VisualGenerationFDLossfor.md` 纳入索引，归入视觉生成方向。
- 为两篇页面补充末尾 `## 索引信息`。
- 将相关图片统一放入 `wiki/assets/` 并更新页面引用。

## [2026-06-05] 摄取 | Embodied Pi 与 GSSM 笔记

- 将 `wiki/Embodied_Pi.md` 纳入索引，归入机器人控制 / VLA 方向。
- 将 `wiki/gssm.md` 纳入索引，归入自动驾驶安全方向。
- 为两篇页面补充末尾 `## 索引信息`。
- 确认 `Embodied_Pi.md` 图片引用指向 `wiki/assets/`。

## [2026-06-05] 维护 | 按 AGENTS 规则整理工程

- 将 `wiki/rl-selfplay.md` 纳入索引，归入自动驾驶安全方向。
- 将 `wiki/rl-selfplay.md` 引用的图片移动到 `wiki/assets/` 并修正引用路径。
- 为 `wiki/rl-selfplay.md` 补充末尾 `## 索引信息`。

## [2026-06-05] 摄取 | SFT、RL 与 OPD 分布视角

- 根据 `raw/assets/sft_rl.md` 中的来源链接生成 `wiki/sft_rl.md`。
- 将 `wiki/sft_rl.md` 纳入索引，新增“大模型后训练”分类。
- 为 `wiki/sft_rl.md` 补充末尾 `## 索引信息`。

## [2026-06-05] 维护 | 用完整 HTML 更新 SFT / RL / OPD 页面

- 根据 `raw/SFT, RL, and On-Policy Distillation Through a Distributional Lens _ wh.html` 更新 `wiki/sft_rl.md`。
- 补充 OPD teacher 实验、RL 少遗忘原因和完整后训练流水线。

## [2026-06-05] 维护 | 为 SFT / RL / OPD 页面补关键图片

- 从完整 HTML 附件中筛选关键图片并复制到 `wiki/assets/`。
- 在 `wiki/sft_rl.md` 中补充分布示意、OPSD token KL、on-policy 几何、MiMo 对比和 reward/entropy 曲线图。

## [2026-06-09] 摄取 | OAT Ordered Action Tokenization

- 根据 `raw/OAT Ordered Action Tokenization.pdf` 生成 `wiki/oat.md`。
- 裁剪并引入 Fig.1、Fig.2、Fig.3、Table I、Table III 和 Table V 等关键图表。
- 将 `wiki/oat.md` 纳入索引，归入机器人控制方向。
- 为 `wiki/oat.md` 补充末尾 `## 索引信息`。

## [2026-06-11] 摄取 | Flow-GRPO 与 Q-Guided Flow 策略

- 根据 `raw/Flow-GRPO Training Flow Matching Models via Online RL.pdf` 生成 `wiki/rl-flow_grpo.md`。
- 根据 `raw/Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning.pdf` 生成 `wiki/rl-test_time_gradient_guidance_flow_policies.md`。
- 裁剪并引入两篇论文的关键图表，统一放入 `wiki/assets/` 并更新页面引用。
- 将两篇页面纳入强化学习与世界模型相关索引，并补充末尾 `## 索引信息`。

## [2026-06-11] 维护 | 补齐索引与流程检查

- 更新 `wiki/index.md`，补入 Flow-GRPO、Q-Guided Flow 和 1000 层无监督 RL 的准确摘要。
- 将 OAT 页面索引从旧文件名 `wiki/oat.md` 修正为 `wiki/EmbodiedOat.md`。
- 加固 `AGENTS.md` 中的摄取流程：新增或整理 wiki 页面后必须复核目标页面、`wiki/index.md` 和 `wiki/log.md`。

## [2026-06-12] 摄取 | GFT 与 RL 基础

- 根据 `raw/GFT From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification.pdf` 生成 `wiki/training-gft.md`。
- 整理 GFT 对 SFT 的 policy gradient 解释、Group Advantage Learning、Dynamic Coefficient Rectification 和 SFT/GFT/GRPO 兼容性结论。
- 裁剪并引入 Fig.1、Fig.2、Table 1/2、Fig.3、Fig.4、Table 3/Fig.5、Table 4/5/Fig.6 等关键图表，统一放入 `wiki/assets/`。
- 新增 `wiki/rl_basic.md`，整理 reward、value、Q、advantage 及常见 RL 方法差异。
- 将 `wiki/training-gft.md` 和 `wiki/rl_basic.md` 纳入 `wiki/index.md` 的“大模型后训练”分类，并补充末尾 `## 索引信息`。
