"""Unit tests for graph modules."""

import os


# 正式图现在在导入时构建模型；测试使用明确的非生产占位 Key，所有外部请求仍被 mock。
os.environ.setdefault("DEEPSEEK_API_KEY", "unit-test-key")
