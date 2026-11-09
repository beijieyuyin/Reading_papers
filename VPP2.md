# Video Prediction Policy 2 (VPP2): Predict Better, Act Better — 论文精读笔记

---

## 1. Motivation & Core Issues (立足问题与成因分析)

### 1.1 核心痛点
世界动作模型（World Action Models, WAMs）依赖“**先预测未来视觉，再反推动作（Inverse Dynamics）**”的范式。但在开放式与分布外（OOD）环境中，现有的 WAM 经常产生**不符合物理常理或违背语言指令的错误视觉预测**，进而导致下游动作执行完全失败。

### 1.2 深层根因
论文指出了现有模型泛化受限的两个本质原因：
1. **基础视频模型的优化目标错位**：主流开源视频基础模型（如 Wan、CogVideoX、SVD）主要面向影视与创意生成，过度追求画面美学质量（Aesthetic Quality），而忽视了**精确的物理交互动力学**和**对操作指令的严格遵循能力**。
2. **动作模块对视频表征的破坏（Catastrophic Interference）**：在已有视频模型中直接接入动作头并联合微调，强监督的动作梯度会严重破坏视频模型已有的开箱即用泛化与时空表征能力。
3. **短程映射的伪相关问题**：传统策略将简短指令与单步观测直接映射到短动作块（Action Chunk），随着数据量扩大，这种一对多的映射极具多模态不确定性，导致策略拟合虚假的局部短程关联。

---

## 2. Background & Paradigm (背景与核心思路)

* **核心假设**：**Predict Better $\to$ Act Better**。只有高质量、强指令跟随的未来视觉预测，才能为底层动作提供稳定、确定的控制先验。
* **基本架构**：
  * **Video Backbone**：Wan2.1-I2V-14B（Flow Matching 视频扩散 Transformer）。
  * **Action Module**：0.9B 参数量 Diffusion Transformer（DiT），采用 Mixture-of-Transformers (MoT) 结构与视频主干解耦交互。
  * **High-Level Planner**：外挂多模态大模型（VLM）处理长程推理与子任务拆解。

---

## 3. Methodology (核心方法与数学建模)

### 3.1 数据工程：去歧义化与标准化管线

为消除视频生成的多模态歧义性，VPP2 建立了严格的自动化数据处理标准：

1. **事件语义切分**：基于速度局部极小值（Local Minima）及末端夹爪开合时刻启发式划分，再经 VLM 校验合并，得到 1–12 秒的语义完整子任务片段。
2. **模板化密集提示词（Dense Video Captioning）**：
   * **四要素模板**：`[执行臂角色]`（明确左/右夹爪，首帧必须可见） + `[操作动作时序]` + `[精确目标物体]`（带空间/外观属性去歧义，首帧必须可见） + `[相机运动与视角配置]`。
3. **统一动作空间与坐标系对齐**：
   对于机械臂数据集 $d$，原始末端位姿为 $T_d = {}^{W_d}T_{E_d} \in SE(3)$。定义工作空间对齐变换矩阵 $A_d = {}^{\bar{W}}T_{W_d}$ 与末端对齐矩阵 $B_d = {}^{E_d}T_{\bar{E}}$，将位姿统一至标准世界系 $\bar{W}$ 与标准末端系 $\bar{E}$：
   $$\bar{T}_d = A_d T_d B_d$$
4. **T 形多视角拼接（T-shape Multi-view）**：固定将主视角置于左侧，右侧上下堆叠两个手腕辅助视角（缺失时以黑色图像填充），尺寸统一为 $416 \times 240$。

---

### 3.2 视频预测模型训练：从事件先验到单步规划

```
[原始视频数据] ──> 【Stage 1: 事件级继续预训练】 (学物理/指令先验, 49帧/变步长)
                         │
                         ▼
                   【Stage 2: 块级后训练 + 一致性蒸馏】 (固定时域8s, 蒸馏为1步推理)
                         │
                         ▼
                   【Stage 3: 动作专家预训练 (MoT)】 (0.9B Action DiT, 冻结视频主干+LoRA)
```

#### Stage 1: 事件级继续预训练（Event-level Continued Pre-training）
* **目标**：在 14B 视频模型中注入操作领域的物理规律与强指令跟随能力。
* **机制**：对整个子任务轨迹 $\tau = (o_0, o_1, \dots, o_T)$ 均匀采样 $N$ 个未来帧：
  $$y_{\text{event}} = \left( o_0, o_{\lfloor T/N \rfloor}, o_{\lfloor 2T/N \rfloor}, \dots, o_T \right)$$
* **优化目标**：采用 Flow Matching 损失：
  $$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{(x_1, c), \epsilon, s} \left[ \| v_\theta(x_s, s, c) - (x_1 - \epsilon) \|_2^2 \right]$$
  其中 $s \in [0, 1]$ 为流动时间，$x_s = (1 - s)\epsilon + s x_1$，$x_1 = \mathcal{E}(y_{\text{event}})$，条件 $c$ 包含子任务详细文本和初帧观测 $o_0$。预测 49 帧，微调 30,000 steps。

#### Stage 2: 块级预测与一致性蒸馏（Chunk-level Post-training & Consistency Distillation）
* **固定时域**：事件级采样导致帧间物理时间间隔不稳定，无法对齐动作。因此强制模型预测固定时间跨度的未来块：**人类操作预测 2 秒，机器人操作预测 8 秒**。设名义采样间隔 $\delta = fH/N$（$H$ 为预测时长，$f$ 为数据集帧率）：
  $$y_{\text{chunk}}^{(t)} = \left( o_t, o_{t+\lfloor \delta \rfloor}, \dots, o_{t+\lfloor N\delta \rfloor} \right)$$
* **一致性蒸馏（CD）**：将多步采样蒸馏为**单步前向推理（1-step generation）**，损失函数为：
  $$\mathcal{L}_{\text{CD}} = \mathbb{E} \left[ \lambda(s, s') \| F_\theta(x_s, s, c) - \text{stopgrad}\left(F_{\bar{\theta}}(\hat{x}_{s'}, s', c)\right) \|_2^2 \right]$$
  蒸馏后，推理阶段只需一次前向传递即可获得未来视觉表征：$x_1 \approx F_\theta(\epsilon, 0, c)$。

---

### 3.3 动作建模：MoT 隐式逆动力学（Stage 3）

* **架构**：采用 0.9B 参数量的 Action DiT，使用 Mixture-of-Transformers (MoT) 架构。
* **抗遗忘策略**：在动作预训练初期，**完全冻结 14B Video DiT 的主干参数，仅通过 LoRA 适配**，避免动作任务的强梯度破坏底层视频表征。
* **时空对齐**：输入 1 步蒸馏出的 8 秒未来视频 Latents（由 Video DiT 生成），Action DiT 仅需 5 步去噪，生成未来 2 秒（$h$ 步）的动作块（Action Chunk）。

---

### 3.4 整体部署与运行流水线 (Deployment & Inference Pipeline)

```
[用户开放式模糊指令: "整理桌面"] 
                 │
                 ▼
     【High-Level: VLM 规划器】
  (拆解为符合模板的子任务描述: "左臂拿起黄色积木放置在收纳盒内...")
                 │
                 ▼
┌───────────────────────────────────────────────┐
│        VPP2 执行闭环 (总延迟 ≈ 0.22s)          │
│                                               │
│  当前观测 o_t ──> 【Video DiT (1-step CD)】    │
│                        │ (约 0.12s)           │
│                        ▼                      │
│                  未来 8s 视频 Latent          │
│                        │                      │
│                        ▼                      │
│                 【Action DiT】                │
│             (MoT 条件注入, 5步去噪)           │
│                        │ (约 0.10s)           │
│                        ▼                      │
│             输出未来 2s 动作块 (Action Chunk)  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
                【机器人底层执行 (ALOHA)】
```

* **延迟优势**：在 `bfloat16` 与 `torch.compile` 下，视频预测耗时 $\sim 0.12\text{s}$，动作去噪耗时 $\sim 0.10\text{s}$，单块推理总延迟仅需 **0.22 秒**，可充分支持闭环控制。

---

## 4. Experiments & Key Findings (实验结果与结论)

### 4.1 视频预测质量与指令跟随能力
* **人类/GPT 联合评测**：在开放式任务中，VPP2-14B 指令跟随成功率达到 **80%（人手）/ 90%（机器人）**，相比 64B 的 Cosmos3-Super-Image2Video 分别超出 10% 与 12%（Cosmos3-64B 为 70% / 78%）。
* **消融实验**：若去除 Stage 1 的事件级预训练、直接做固定步长预测（VPP2-Fixed-Step），指令跟随率直接从 90% 断崖式跌落至 42%，证明**先用完整事件建立语义到轨迹的确定性映射，是泛化性的决定性前提**。

### 4.2 真实机械臂零样本操作（Zero-Shot on ALOHA）
* 在 10 类未见任务（包含 Stack, Pick, Place, Fold, Pour, Open 等）的真实 ALOHA 机械臂零样本测试中：
  * **VPP2 平均成功率：58.5%**
  * $\pi_{0.5}$ (Physical Intelligence)：40.0%
  * Fast-WAM-14B：20.5%
* VPP2 在 9/10 类任务中均大幅领先，展现出对未见物体几何、视角扰动极强的零样本适应性。

### 4.3 泛化基准评测（LIBERO & RoboDojo）
* **LIBERO-Pro**（空间与任务扰动）：
  * VPP2 达到 **45.0%** 成功率（强基线 $\pi_{0.5}$ 仅 11.0%，其他 VLA 模型几乎全部降至 6% 以下）。在 Task 扰动测试中达到 47.8%（基线仅 6.8%），证明其真正理解了语义指令，而非机械记忆轨迹。
* **LIBERO-OOD**（物体/目标组合泛化）：
  * VPP2 达到 **63.9%**，远超 Cosmos-Policy（10.5%）与 Temporal Ratio（59.4%）。
* **RoboDojo 仿真基准**：
  * 评分达到 **39.26**，成功率 **32.26%**（登顶 Leaderboard SOTA），超越 GPT-6-Astra（28.97 / 22.48%）与 Xiaomi-Robotics-1（20.07 / 13.93%）。
  * 引入 VLM 动态子任务规划后，在五项挑战性长程任务上平均成功率由 27.6% 跃升至 **57.6%**（Tic-tac-toe 从 0% 提升至 38%）。

---

## 5. Insights & Contributions (总结与启发)

1. **确立了 WAM 领域的核心范式**：以往工作倾向于在架构上堆叠花哨的 Video-Action 交互模块，VPP2 用实验证明：**泛化的瓶颈在视频端，而非动作解码端**。
2. **两阶段时空解耦的训练策略**：
   * *阶段一（Event-level）*：利用长程、完整的事件视频 + 极度详尽的提示词压制多模态发散性，教会模型正确的物理交互先验；
   * *阶段二（Chunk-level + CD）*：通过一致性蒸馏锁定统一的时序尺度并加速，将生成模型转变为实时“视觉规划器”。
3. **参数保护机制**：动作预训练早期冻结视频主干、仅用 LoRA 适配，解决了长期以来“动作梯度破坏预训练多模态先验”的痼疾。