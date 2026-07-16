"""WiFi Agent 数据处理 Pipeline。

从原始JSON数据（8821条）到训练数据（parquet）的完整处理链路：

阶段1: 数据分级筛选（stage1_filter.py）
阶段2: 工具名映射（stage2_tool_mapping.py）
阶段3: 四件套转换（stage3_convert_to_quartet.py）
阶段4: 场景分类（stage4_classify.py）
阶段5: 训练/测试划分（stage5_split.py）
阶段6: SFT训练格式构建（stage6_build_sft_parquet.py）
阶段7: 数据增强（stage7_enhance.py）

一键运行: python data_pipeline/run_all.py
"""
