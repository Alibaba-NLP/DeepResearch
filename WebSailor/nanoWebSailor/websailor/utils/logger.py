#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块

提供统一的日志配置和管理功能。

主要函数：
- setup_logger: 设置日志器

作者: Evan Zuo
日期: 2025年1月
"""

import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any


def setup_logger(
    level: str = "INFO",
    format_str: Optional[str] = None,
    file_path: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "7 days",
    colorize: bool = True
) -> None:
    """设置日志器配置
    
    Args:
        level: 日志级别
        format_str: 日志格式字符串
        file_path: 日志文件路径
        rotation: 日志轮转大小
        retention: 日志保留时间
        colorize: 是否启用颜色
    """
    # 移除默认处理器
    logger.remove()
    
    # 默认格式
    if format_str is None:
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=colorize,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件处理器
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            file_path,
            format=format_str,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
        
        logger.info(f"日志文件设置为: {file_path}")
    
    logger.info(f"日志器配置完成，级别: {level}")


def get_logger(name: str, log_path: str = None):
    """获取命名日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    if log_path:
        logger.add(log_path)
        logger.info(f"Log file is save to {log_path}")
    return logger.bind(name=name)


class LoggerContextManager:
    """日志上下文管理器
    
    用于临时修改日志级别或添加额外的处理器。
    """
    
    def __init__(
        self,
        level: Optional[str] = None,
        extra_handlers: Optional[Dict[str, Any]] = None
    ):
        """初始化上下文管理器
        
        Args:
            level: 临时日志级别
            extra_handlers: 额外的处理器配置
        """
        self.level = level
        self.extra_handlers = extra_handlers or {}
        self.original_handlers = []
        self.added_handler_ids = []
    
    def __enter__(self):
        """进入上下文"""
        # 保存当前处理器
        self.original_handlers = logger._core.handlers.copy()
        
        # 设置临时级别
        if self.level:
            for handler_id in logger._core.handlers:
                logger._core.handlers[handler_id].levelno = logger.level(self.level).no
        
        # 添加额外处理器
        for name, config in self.extra_handlers.items():
            handler_id = logger.add(**config)
            self.added_handler_ids.append(handler_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 移除添加的处理器
        for handler_id in self.added_handler_ids:
            logger.remove(handler_id)
        
        # 恢复原始处理器设置
        if self.level:
            logger._core.handlers = self.original_handlers 