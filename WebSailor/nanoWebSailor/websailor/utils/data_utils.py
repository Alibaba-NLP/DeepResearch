#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具模块

提供数据处理和信息混淆相关的工具函数。

主要类：
- DataProcessor: 数据处理器
- ObfuscationEngine: 信息混淆引擎

作者: Evan Zuo
日期: 2025年1月
"""

import re
import random
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from loguru import logger


class DataProcessor:
    """数据处理器
    
    提供数据清洗、转换、格式化等功能。
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 统一标点符号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('…', '...')
        
        return text
    
    @staticmethod
    def normalize_date(
        date_str: str,
        input_formats: Optional[List[str]] = None
    ) -> Optional[datetime]:
        """标准化日期
        
        Args:
            date_str: 日期字符串
            input_formats: 输入格式列表
            
        Returns:
            datetime对象
        """
        if input_formats is None:
            input_formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%Y.%m.%d',
                '%Y年%m月%d日',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%B %d, %Y',
                '%b %d, %Y'
            ]
        
        for fmt in input_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def format_number(
        number: Union[int, float],
        precision: int = 2,
        use_thousands_separator: bool = True
    ) -> str:
        """格式化数字
        
        Args:
            number: 数字
            precision: 小数位数
            use_thousands_separator: 是否使用千位分隔符
            
        Returns:
            格式化后的字符串
        """
        try:
            if isinstance(number, int):
                formatted = format(number, ',d' if use_thousands_separator else 'd')
            else:
                formatted = format(number, f',.{precision}f' if use_thousands_separator else f'.{precision}f')
            return formatted
        except Exception:
            return str(number)
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """提取文本中的数字
        
        Args:
            text: 输入文本
            
        Returns:
            数字列表
        """
        pattern = r'-?\d*\.?\d+'
        matches = re.finditer(pattern, text)
        return [float(match.group()) for match in matches]
    
    @staticmethod
    def safe_json_loads(
        json_str: str,
        default: Any = None
    ) -> Any:
        """安全解析JSON
        
        Args:
            json_str: JSON字符串
            default: 解析失败时的默认值
            
        Returns:
            解析结果
        """
        try:
            return json.loads(json_str)
        except Exception:
            return default
    
    @staticmethod
    def safe_json_dumps(
        obj: Any,
        default: str = "{}",
        ensure_ascii: bool = False,
        indent: Optional[int] = None
    ) -> str:
        """安全序列化JSON
        
        Args:
            obj: Python对象
            default: 序列化失败时的默认值
            ensure_ascii: 是否确保ASCII输出
            indent: 缩进空格数
            
        Returns:
            JSON字符串
        """
        try:
            return json.dumps(
                obj,
                ensure_ascii=ensure_ascii,
                indent=indent
            )
        except Exception:
            return default


class ObfuscationEngine:
    """信息混淆引擎
    
    实现各种信息混淆策略，用于生成具有不确定性的训练数据。
    """
    
    def __init__(self):
        """初始化混淆引擎"""
        # 时间混淆参数
        self.date_formats = [
            '%Y年代初',
            '%Y年代中期',
            '%Y年代末',
            '%Y年前后',
            '约%Y年',
            '%Y年左右'
        ]
        
        # 数值混淆参数
        self.number_formats = [
            '约{0}',
            '大约{0}',
            '接近{0}',
            '{0}左右',
            '大概{0}'
        ]
        
        # 实体混淆参数
        self.entity_patterns = [
            '一位{0}',
            '某{0}',
            '一名{0}',
            '一个{0}'
        ]
        
        logger.info("信息混淆引擎初始化完成")
    
    def obfuscate_date(
        self,
        date: Union[str, datetime],
        precision: str = 'year',
        format_style: str = 'formal'
    ) -> str:
        """混淆日期信息
        
        Args:
            date: 日期
            precision: 精度（year/decade/century）
            format_style: 格式风格（formal/casual）
            
        Returns:
            混淆后的日期字符串
        """
        if isinstance(date, str):
            date = DataProcessor.normalize_date(date)
            if date is None:
                return "未知时间"
        
        try:
            if precision == 'decade':
                decade = (date.year // 10) * 10
                if format_style == 'formal':
                    return f"{decade}年代"
                else:
                    return random.choice([
                        f"{decade}年代初",
                        f"{decade}年代中期",
                        f"{decade}年代末"
                    ])
            
            elif precision == 'century':
                century = (date.year // 100) + 1
                return f"第{century}世纪"
            
            else:  # year
                if format_style == 'formal':
                    return str(date.year)
                else:
                    year_format = random.choice(self.date_formats)
                    return datetime.strftime(date, year_format)
                    
        except Exception:
            return "未知时间"
    
    def obfuscate_number(
        self,
        number: Union[int, float],
        variance: float = 0.1,
        style: str = 'approximate'
    ) -> str:
        """混淆数值信息
        
        Args:
            number: 数值
            variance: 变化幅度
            style: 混淆风格（approximate/range/order）
            
        Returns:
            混淆后的数值字符串
        """
        try:
            if style == 'range':
                # 生成范围
                lower = number * (1 - variance)
                upper = number * (1 + variance)
                return f"{DataProcessor.format_number(lower)}至{DataProcessor.format_number(upper)}"
            
            elif style == 'order':
                # 数量级表示
                magnitude = 10 ** int(np.log10(number))
                return f"数{magnitude}"
            
            else:  # approximate
                # 近似值
                variance_amount = number * variance
                approximate = number + random.uniform(-variance_amount, variance_amount)
                formatted = DataProcessor.format_number(approximate)
                return random.choice(self.number_formats).format(formatted)
                
        except Exception:
            return str(number)
    
    def obfuscate_entity(
        self,
        entity: str,
        entity_type: str,
        style: str = 'anonymous'
    ) -> str:
        """混淆实体信息
        
        Args:
            entity: 实体名称
            entity_type: 实体类型
            style: 混淆风格（anonymous/partial/role）
            
        Returns:
            混淆后的实体描述
        """
        try:
            if style == 'anonymous':
                # 完全匿名化
                return random.choice(self.entity_patterns).format(entity_type)
            
            elif style == 'partial':
                # 部分信息保留
                if len(entity) <= 2:
                    return f"{entity[0]}某"
                else:
                    return f"{entity[0]}某{entity[-1]}"
            
            elif style == 'role':
                # 基于角色的描述
                return f"这位{entity_type}"
            
            else:
                return entity
                
        except Exception:
            return entity
    
    def obfuscate_location(
        self,
        location: str,
        precision: str = 'city',
        style: str = 'formal'
    ) -> str:
        """混淆地理位置信息
        
        Args:
            location: 地理位置
            precision: 精度（city/region/country）
            style: 混淆风格（formal/descriptive）
            
        Returns:
            混淆后的位置描述
        """
        location_patterns = {
            'city': [
                '某座城市',
                '一座城市',
                '某市'
            ],
            'region': [
                '某个地区',
                '一个区域',
                '某区域'
            ],
            'country': [
                '某个国家',
                '一个国家',
                '某国'
            ]
        }
        
        try:
            if style == 'descriptive':
                return random.choice(location_patterns[precision])
            else:
                return f"某{precision}"
        except Exception:
            return location
    
    def generate_uncertainty(self, text: str, level: int = 1) -> str:
        """生成不确定性表述
        
        Args:
            text: 输入文本
            level: 不确定性级别（1-3）
            
        Returns:
            添加不确定性后的文本
        """
        uncertainty_patterns = {
            1: [  # 低度不确定
                '可能{0}',
                '似乎{0}',
                '大概{0}'
            ],
            2: [  # 中度不确定
                '据说{0}',
                '有传言称{0}',
                '可能会{0}'
            ],
            3: [  # 高度不确定
                '有些人认为{0}',
                '存在一种说法{0}',
                '不能确定是否{0}'
            ]
        }
        
        try:
            level = max(1, min(level, 3))  # 限制在1-3之间
            pattern = random.choice(uncertainty_patterns[level])
            return pattern.format(text)
        except Exception:
            return text 