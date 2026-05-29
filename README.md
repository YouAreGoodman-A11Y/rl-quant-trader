# 🤖 强化学习量化交易架构 (RL Quant Trader)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RL Framework](https://img.shields.io/badge/RL-Stable_Baselines3-brightgreen)](#)
[![Status](https://img.shields.io/badge/Status-Active_Research-yellow)](#)

这是一个专注于探索**强化学习 (Reinforcement Learning)** 在多因子量化选股与动态仓位控制中应用的实验性架构。

---

## 📌 研究背景与核心痛点

传统多因子模型（如 XGBoost、LightGBM 或线性回归）主要解决的是“预测横截面收益（Alpha）”的问题，但往往忽略了以下痛点：
1. **组合构建（Portfolio Construction）的割裂**：预测出了分数，如何分配资金？
2. **交易摩擦（Friction）**：高换手带来的滑点和手续费往往吃掉大量账面收益。
3. **宏观避险**：在系统性风险（如大盘主跌浪）来临时，传统选股模型缺乏主动的仓位管控能力。

本仓库致力于使用强化学习（如 PPO 算法），将“因子预测”与“交易执行”融合在一个端到端（End-to-End）的动作空间中。

---

## 🧠 动作空间演进 (Action Space Evolution)

本架构的动作空间（Action Space）在不断迭代，遵循“单点放开，控制变量”的纪律：
*   **V4 阶段**：仅开放 `Cash_Ratio`，控制整体现金水位，选股定死在头部。
*   **V5 阶段**：仅开放 `Sell_Rank_Pct`，控制弱势股汰除容忍度，满仓买入。
*   **V6 阶段**：开放 `Buy_Start_Pct`，实现激进/防御区间动态滑窗。

> **⚠️ 实盘纪律反思**：
> 在引入换手率惩罚和最大回撤惩罚（DD Penalty）时，需警惕模型陷入“Collapse to Zero（彻底空仓以规避换手摩擦）”或“Collapse to Max（在无约束下无脑满仓）”的安全陷阱。

---

## 🛡️ 架构规范与隔离机制

1. **训练与回测物理隔离**：严禁将模型训练（Training）与评估出图（Evaluation / Backtesting）糅合在同一个脚本中，以规避评估阶段的环境越界截断问题（Off-by-One Error）。
2. **极速矩阵化（Vectorization）**：环境步进 `step()` 必须使用纯 Numpy 布尔切片，禁止在循环内使用 Pandas DataFrame 操作，确保训练速度。
3. **数据隔离（Data Privacy）**：
   * 本仓库仅托管强化学习环境 (`Gym Env`) 与算法代码。
   * 预训练生成的权重模型文件（如 `.zip`）以及海量特征数据 (`.parquet`) 均已被 `.gitignore` 拦截，请在本地集群中运行与存储。
