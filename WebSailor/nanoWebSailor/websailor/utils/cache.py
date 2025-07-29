#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理模块

提供统一的缓存管理接口，支持内存缓存、持久化缓存等。

主要类：
- CacheManager: 缓存管理器

作者: Evan Zuo
日期: 2025年1月
"""

import pickle
import time
import hashlib
from typing import Any, Optional, Dict, Set
from pathlib import Path
from threading import RLock
from dataclasses import dataclass
from loguru import logger


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class CacheManager:
    """缓存管理器
    
    提供多级缓存功能，支持内存缓存和磁盘持久化。
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600,
        cache_dir: Optional[str] = None,
        enable_persistence: bool = False
    ):
        """初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
            cache_dir: 持久化缓存目录
            enable_persistence: 是否启用持久化
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        
        # 内存缓存
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []  # LRU访问顺序
        self._lock = RLock()
        
        # 持久化设置
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        logger.info(f"缓存管理器初始化完成，最大条目数: {max_size}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        with self._lock:
            # 检查内存缓存
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    # 更新访问顺序
                    self._update_access_order(key)
                    return entry.value
                else:
                    # 删除过期条目
                    self._remove_entry(key)
            
            # 检查持久化缓存
            if self.enable_persistence and self.cache_dir:
                try:
                    cache_file = self._get_cache_file(key)
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            entry = pickle.load(f)
                        
                        if not entry.is_expired():
                            # 加载到内存缓存
                            self._set_memory_cache(key, entry)
                            return entry.value
                        else:
                            # 删除过期的持久化文件
                            cache_file.unlink(missing_ok=True)
                
                except Exception as e:
                    logger.warning(f"读取持久化缓存失败 {key}: {e}")
            
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示使用默认TTL
        """
        if ttl is None:
            ttl = self.default_ttl
        
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl
        )
        
        with self._lock:
            # 设置内存缓存
            self._set_memory_cache(key, entry)
            
            # 持久化缓存
            if self.enable_persistence and self.cache_dir:
                try:
                    cache_file = self._get_cache_file(key)
                    with open(cache_file, 'wb') as f:
                        pickle.dump(entry, f)
                except Exception as e:
                    logger.warning(f"持久化缓存失败 {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """删除缓存条目
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            # 删除内存缓存
            memory_deleted = self._remove_entry(key)
            
            # 删除持久化缓存
            persistence_deleted = False
            if self.enable_persistence and self.cache_dir:
                try:
                    cache_file = self._get_cache_file(key)
                    if cache_file.exists():
                        cache_file.unlink()
                        persistence_deleted = True
                except Exception as e:
                    logger.warning(f"删除持久化缓存失败 {key}: {e}")
            
            return memory_deleted or persistence_deleted
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            # 清空内存缓存
            self._cache.clear()
            self._access_order.clear()
            
            # 清空持久化缓存
            if self.enable_persistence and self.cache_dir:
                try:
                    for cache_file in self.cache_dir.glob("*.cache"):
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"清空持久化缓存失败: {e}")
        
        logger.info("缓存已清空")
    
    def cleanup_expired(self) -> int:
        """清理过期缓存
        
        Returns:
            清理的条目数量
        """
        expired_keys = []
        
        with self._lock:
            # 检查内存缓存
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # 删除过期条目
            for key in expired_keys:
                self._remove_entry(key)
            
            # 清理持久化缓存
            if self.enable_persistence and self.cache_dir:
                try:
                    for cache_file in self.cache_dir.glob("*.cache"):
                        try:
                            with open(cache_file, 'rb') as f:
                                entry = pickle.load(f)
                            if entry.is_expired():
                                cache_file.unlink()
                                expired_keys.append(cache_file.stem)
                        except Exception:
                            # 损坏的文件也删除
                            cache_file.unlink()
                            expired_keys.append(cache_file.stem)
                except Exception as e:
                    logger.warning(f"清理持久化缓存失败: {e}")
        
        if expired_keys:
            logger.info(f"清理了{len(expired_keys)}个过期缓存条目")
        
        return len(expired_keys)
    
    def _set_memory_cache(self, key: str, entry: CacheEntry) -> None:
        """设置内存缓存（内部方法）"""
        # 如果缓存已满，删除最少使用的条目
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                oldest_key = self._access_order[0]
                self._remove_entry(oldest_key)
        
        self._cache[key] = entry
        self._update_access_order(key)
    
    def _update_access_order(self, key: str) -> None:
        """更新访问顺序（内部方法）"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove_entry(self, key: str) -> bool:
        """删除缓存条目（内部方法）"""
        removed = False
        if key in self._cache:
            del self._cache[key]
            removed = True
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        return removed
    
    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径（内部方法）"""
        # 使用key的哈希值作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            memory_size = len(self._cache)
            
            # 统计持久化缓存
            persistence_size = 0
            if self.enable_persistence and self.cache_dir:
                try:
                    persistence_size = len(list(self.cache_dir.glob("*.cache")))
                except Exception:
                    pass
            
            return {
                'memory_size': memory_size,
                'persistence_size': persistence_size,
                'max_size': self.max_size,
                'memory_usage_ratio': memory_size / self.max_size if self.max_size > 0 else 0,
                'default_ttl': self.default_ttl,
                'enable_persistence': self.enable_persistence,
            }
    
    def __contains__(self, key: str) -> bool:
        """支持in操作符"""
        return self.get(key) is not None
    
    def __len__(self) -> int:
        """返回缓存条目数量"""
        with self._lock:
            return len(self._cache) 