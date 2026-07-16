"""WiFi Agent 训练 Pipeline。

SFT训练: training/sft_train.py（transformers Trainer + DeepSpeed）
RL训练:  training/rl_train_simple.py（简化版REINFORCE）
评测:    training/evaluate.py（AgentLoop + Verifier）

启动SFT: bash training/launch_sft.sh
"""
