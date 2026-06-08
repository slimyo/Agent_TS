"""工业少样本多任务自适应规划系统（multi-agent）。

设计蒸馏自 research/method1-5 + finish1-5 的全部经验教训：
  - Curator 诊断（method1/2）
  - Saturation-aware 规划（method5 / F-R9.7-10.3：先判该不该 route）
  - N-conditional fallback + honest abstain（method3 v10/B7v2 / F-R8.x）
  - 无数据泄漏（method3 M9 / LODO）
  - 可解释 trace（method3 M8 attribution 精神）
覆盖三大任务：Forecasting / Classification / Anomaly-Detection。
"""
