#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor数据合成模块

该模块实现了论文中SailorFog-QA数据集的完整合成管道，用于生成具有高不确定性且难以降低的
Level 3复杂任务。主要包括以下组件：

- EntityGenerator: 使用大模型生成稀有实体
- GraphBuilder: 基于实体构建复杂知识图谱
- SubgraphSampler: 从知识图谱中采样不同拓扑的子图
- QAGenerator: 基于子图生成问答对
- Obfuscator: 通过信息混淆技术增加任务难度

核心思想：
通过构建具有复杂、非线性结构的知识图谱，并结合信息混淆技术，
生成需要多步推理和创造性探索的高难度任务，从而训练智能体的超人推理能力。

数据生成流程：
1. 稀有实体生成 (EntityGenerator)
2. 知识图谱构建 (GraphBuilder)
3. 子图采样 (SubgraphSampler)
4. 问答对生成 (QAGenerator)
5. 信息混淆 (Obfuscator)

作者: Evan Zuo
日期: 2025年1月
"""

from .entity_generator import EntityGenerator, GeneratedEntity
from .graph_builder import GraphBuilder, GraphNode, GraphEdge
from .subgraph_sampler import SubgraphSampler
from .qa_generator import QAGenerator
from .graph_expander import GraphExpander
from .mock_web_search import MockWebSearch
from .triplet_extractor import TripleExtractor
# from .obfuscator import Obfuscator  # TODO: Implement this module

__all__ = [
    # 实体生成器
    "EntityGenerator",
    "GeneratedEntity",
    
    # 图构建器
    "GraphBuilder",
    "GraphNode", 
    "GraphEdge",
    
    # 子图采样器
    "SubgraphSampler",
    
    # QA生成器
    "QAGenerator",

    # 图扩展器
    "GraphExpander",

    # 模拟网页搜索
    "MockWebSearch",

    # 三元组抽取
    "TripleExtractor",
] 