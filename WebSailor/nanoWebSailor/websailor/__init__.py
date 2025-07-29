#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor: 超人推理的Web智能体框架

WebSailor是一个专注于构建具备超人推理能力的Web智能体的开源框架。
本框架实现了论文《WebSailor: Navigating Super-human Reasoning for Web Agent》中的核心方法。

主要模块：
- data_synthesis: 复杂训练数据合成，包括SailorFog-QA数据集生成
- trajectory: 专家轨迹处理和推理重构
- agent: ReAct智能体核心实现
- evaluation: 多基准评估框架
- utils: 通用工具和配置管理

作者: Evan Zuo
日期: 2025年1月
版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Evan Zuo"
__email__ = "zuoyifan132@@gmail.com"
__description__ = "WebSailor: 具备超人推理能力的Web智能体框架"

# 导入主要模块
from . import data_synthesis
from . import trajectory
from . import agent
from . import utils

# 导入核心类和函数
from .agent.react_agent import ReActAgent

__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    
    # 主要模块
    "data_synthesis",
    "trajectory", 
    "agent",
    "utils",
    
    # 核心类
    "ReActAgent",
    "KnowledgeGraphBuilder", 
] 