#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

提供统一的配置管理接口，支持从YAML文件、环境变量等多种来源加载配置。

主要类：
- Config: 主配置管理类

作者: Evan Zuo
日期: 2025年1月
"""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from loguru import logger


class Config:
    """配置管理器
    
    支持从多种来源加载和管理配置，包括YAML文件、环境变量等。
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self._config: DictConfig = OmegaConf.create({})
        self._load_default_config()
        
        if config_path:
            self.load_from_file(config_path)
        
        # 从环境变量覆盖配置
        self._load_from_env()
        
        logger.info("配置管理器初始化完成")

    def print_config(self, prefix: str = "", indent: int = 0, section_filter: Optional[str] = None) -> None:
        """格式化打印配置参数
        
        Args:
            prefix: 配置键前缀，用于内部递归调用
            indent: 缩进级别，用于内部递归调用
            section_filter: 可选，只打印指定的配置部分(如 'agent'、'logging' 等)
        """
        config_dict = self.to_dict()
        
        def _print_config_recursive(d, prefix="", indent=0):
            indent_str = "  " * indent
            for key, value in sorted(d.items()):
                current_key = f"{prefix}.{key}" if prefix else key
                
                # 如果设置了section_filter，则只打印匹配的部分
                if section_filter and not current_key.startswith(section_filter):
                    continue
                    
                if isinstance(value, dict):
                    logger.info(f"{indent_str}{key}:")
                    _print_config_recursive(value, current_key, indent + 1)
                else:
                    # 特殊处理列表类型，使其更易读
                    if isinstance(value, list):
                        if len(value) > 0 and not isinstance(value[0], dict):
                            value_str = ", ".join(str(v) for v in value)
                            logger.info(f"{indent_str}{key}: [{value_str}]")
                        else:
                            logger.info(f"{indent_str}{key}: [")
                            for i, item in enumerate(value):
                                logger.info(f"{indent_str}  {i}:")
                                if isinstance(item, dict):
                                    _print_config_recursive(item, "", indent + 2)
                                else:
                                    logger.info(f"{indent_str}    {item}")
                            logger.info(f"{indent_str}]")
                    else:
                        logger.info(f"{indent_str}{key}: {value}")
        
        # 打印标题
        if section_filter:
            logger.info(f"=== 配置参数 [{section_filter}] ===")
        else:
            logger.info("=== 配置参数 ===")
        
        _print_config_recursive(config_dict, prefix, indent)
        logger.info("=" * 20)

    def _load_default_config(self) -> None:
        """加载默认配置"""
        default_config = {
            # 数据合成配置
            'data_synthesis': {
                'batch_size': 32,
                'max_qa_pairs': 1000,
                'obfuscation_rate': 0.3,
                'complexity_level': 3,
            },
            
            # 知识图谱配置
            'graph': {
                'max_nodes': 100,
                'max_edges': 200,
                'expansion_probability': 0.7,
                'revisit_probability': 0.3,
            },
            
            # Wikidata配置
            'wikidata': {
                'request_timeout': 30,
                'rate_limit_delay': 1.0,
                'max_retries': 3,
            },
            
            # 轨迹生成配置
            'trajectory': {
                'max_steps': 30,
                'expert_model': 'QwQ-32B',
                'reconstruct_thoughts': True,
                'short_cot': True,
            },
            
            # 智能体配置
            'agent': {
                'model_name': 'gpt-4',
                'temperature': 0.1,
                'max_tokens': 2048,
                'tools': ['search', 'browse', 'extract'],
            },
            
            # 评估配置
            'evaluation': {
                'benchmarks': ['browsecomp-en', 'browsecomp-zh', 'gaia', 'xbench'],
                'metrics': ['pass_at_k', 'success_rate', 'reasoning_quality'],
                'k_values': [1, 3, 5],
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',
                'format': '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
                'file_path': 'logs/websailor.log',
            },
            
            # 缓存配置
            'cache': {
                'enable': True,
                'ttl': 3600,  # 1小时
                'max_size': 1000,
            },
            
            # Web配置
            'web': {
                'user_agent': 'WebSailor/1.0',
                'request_timeout': 30,
                'retry_attempts': 3,
                'headless': True,
            }
        }
        
        self._config = OmegaConf.create(default_config)
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """从文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        config_path = rf"{config_path}"
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                file_config = OmegaConf.load(config_path)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 合并配置
            self._config = OmegaConf.merge(self._config, file_config)
            logger.info(f"已加载配置文件: {config_path}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        env_mapping = {
            'WEBSAILOR_LOG_LEVEL': 'logging.level',
            'WEBSAILOR_CACHE_ENABLE': 'cache.enable',
            'WEBSAILOR_AGENT_MODEL': 'agent.model_name',
            'WEBSAILOR_BATCH_SIZE': 'data_synthesis.batch_size',
            'WEBSAILOR_MAX_NODES': 'graph.max_nodes',
        }
        
        for env_key, config_key in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # 类型转换
                if env_key in ['WEBSAILOR_CACHE_ENABLE']:
                    env_value = env_value.lower() in ['true', '1', 'yes']
                elif env_key in ['WEBSAILOR_BATCH_SIZE', 'WEBSAILOR_MAX_NODES']:
                    env_value = int(env_value)
                
                OmegaConf.set(self._config, config_key, env_value)
                logger.debug(f"从环境变量设置配置: {config_key} = {env_value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键（如 'agent.model_name'）
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default
        
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        # 使用 OmegaConf.update 来设置值
        update_dict = {}
        keys = key.split('.')
        current = update_dict
        for k in keys[:-1]:
            current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        
        update_config = OmegaConf.create(update_dict)
        self._config = OmegaConf.merge(self._config, update_config)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            config_dict: 配置字典
        """
        update_config = OmegaConf.create(config_dict)
        self._config = OmegaConf.merge(self._config, update_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return OmegaConf.to_container(self._config, resolve=True)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """保存配置到文件
        
        Args:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(self._config, f)
        
        logger.info(f"配置已保存到: {output_path}")
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持in操作符"""
        return OmegaConf.select(self._config, key) is not None 