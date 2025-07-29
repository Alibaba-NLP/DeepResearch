#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web工具模块

提供网页抓取和浏览器管理功能，支持智能体的Web交互需求。

主要类：
- WebScraper: 网页抓取器
- BrowserManager: 浏览器管理器

作者: Evan Zuo
日期: 2025年1月
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from loguru import logger

from .cache import CacheManager


class WebScraper:
    """网页抓取器
    
    提供异步网页抓取和内容提取功能。
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_timeout: int = 30,
        retry_attempts: int = 3,
        enable_cache: bool = True
    ):
        """初始化抓取器
        
        Args:
            cache_dir: 缓存目录
            user_agent: User-Agent字符串
            request_timeout: 请求超时时间（秒）
            retry_attempts: 重试次数
            enable_cache: 是否启用缓存
        """
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.user_agent = user_agent or 'WebSailor/1.0'
        
        # 初始化缓存
        if enable_cache:
            self.cache = CacheManager(
                cache_dir=cache_dir,
                enable_persistence=True
            )
        else:
            self.cache = None
        
        # 初始化会话
        self._session = None
        logger.info("网页抓取器初始化完成")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={'User-Agent': self.user_agent}
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def get_page(
        self,
        url: str,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> Optional[str]:
        """获取网页内容
        
        Args:
            url: 目标URL
            use_cache: 是否使用缓存
            cache_ttl: 缓存过期时间
            
        Returns:
            网页HTML内容
        """
        # 检查缓存
        if use_cache and self.cache:
            cached_content = self.cache.get(url)
            if cached_content:
                logger.debug(f"从缓存获取页面: {url}")
                return cached_content
        
        # 发起请求
        for attempt in range(self.retry_attempts):
            try:
                async with self._session.get(
                    url,
                    timeout=self.request_timeout
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # 更新缓存
                        if use_cache and self.cache:
                            self.cache.set(url, content, ttl=cache_ttl)
                        
                        return content
                    else:
                        logger.warning(
                            f"请求失败 {url}: HTTP {response.status}"
                        )
            
            except asyncio.TimeoutError:
                logger.warning(
                    f"请求超时 {url} (尝试 {attempt + 1}/{self.retry_attempts})"
                )
            except Exception as e:
                logger.warning(
                    f"请求异常 {url} (尝试 {attempt + 1}/{self.retry_attempts}): {e}"
                )
            
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    async def extract_links(
        self,
        url: str,
        content: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> List[str]:
        """提取页面链接
        
        Args:
            url: 页面URL
            content: 页面内容，如果为None则先获取页面
            base_url: 基础URL，用于相对链接转绝对链接
            
        Returns:
            链接列表
        """
        if content is None:
            content = await self.get_page(url)
            if content is None:
                return []
        
        if base_url is None:
            base_url = url
        
        links = []
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href:
                    # 转换为绝对URL
                    absolute_url = urljoin(base_url, href)
                    if self._is_valid_url(absolute_url):
                        links.append(absolute_url)
        except Exception as e:
            logger.warning(f"提取链接失败 {url}: {e}")
        
        return list(set(links))  # 去重
    
    async def extract_text(
        self,
        url: str,
        content: Optional[str] = None,
        clean: bool = True
    ) -> str:
        """提取页面文本
        
        Args:
            url: 页面URL
            content: 页面内容，如果为None则先获取页面
            clean: 是否清理文本
            
        Returns:
            页面文本内容
        """
        if content is None:
            content = await self.get_page(url)
            if content is None:
                return ""
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            if clean:
                # 清理空白字符
                lines = (line.strip() for line in text.splitlines())
                text = ' '.join(line for line in lines if line)
            
            return text
            
        except Exception as e:
            logger.warning(f"提取文本失败 {url}: {e}")
            return ""
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效（内部方法）"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class BrowserManager:
    """浏览器管理器
    
    管理Selenium WebDriver实例，提供浏览器自动化功能。
    """
    
    def __init__(
        self,
        browser_type: str = "chrome",
        headless: bool = True,
        implicit_wait: int = 10,
        page_load_timeout: int = 30,
        download_dir: Optional[str] = None
    ):
        """初始化浏览器管理器
        
        Args:
            browser_type: 浏览器类型（chrome/firefox）
            headless: 是否使用无头模式
            implicit_wait: 隐式等待时间
            page_load_timeout: 页面加载超时时间
            download_dir: 下载目录
        """
        self.browser_type = browser_type.lower()
        self.headless = headless
        self.implicit_wait = implicit_wait
        self.page_load_timeout = page_load_timeout
        self.download_dir = download_dir
        
        self._driver = None
        logger.info(f"浏览器管理器初始化完成，类型: {browser_type}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
    
    def start(self) -> None:
        """启动浏览器"""
        if self._driver is not None:
            return
        
        try:
            if self.browser_type == "chrome":
                options = webdriver.ChromeOptions()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                if self.download_dir:
                    prefs = {
                        "download.default_directory": str(Path(self.download_dir).absolute()),
                        "download.prompt_for_download": False
                    }
                    options.add_experimental_option("prefs", prefs)
                
                self._driver = webdriver.Chrome(options=options)
                
            elif self.browser_type == "firefox":
                options = webdriver.FirefoxOptions()
                if self.headless:
                    options.add_argument('--headless')
                
                if self.download_dir:
                    options.set_preference("browser.download.folderList", 2)
                    options.set_preference("browser.download.dir", 
                                        str(Path(self.download_dir).absolute()))
                    options.set_preference("browser.download.useDownloadDir", True)
                
                self._driver = webdriver.Firefox(options=options)
            
            else:
                raise ValueError(f"不支持的浏览器类型: {self.browser_type}")
            
            # 设置超时
            self._driver.implicitly_wait(self.implicit_wait)
            self._driver.set_page_load_timeout(self.page_load_timeout)
            
            logger.info(f"浏览器已启动: {self.browser_type}")
            
        except Exception as e:
            logger.error(f"启动浏览器失败: {e}")
            raise
    
    def stop(self) -> None:
        """停止浏览器"""
        if self._driver:
            try:
                self._driver.quit()
            except Exception as e:
                logger.warning(f"关闭浏览器失败: {e}")
            finally:
                self._driver = None
    
    def navigate(self, url: str) -> bool:
        """导航到指定URL
        
        Args:
            url: 目标URL
            
        Returns:
            是否成功导航
        """
        if not self._driver:
            self.start()
        
        try:
            self._driver.get(url)
            return True
        except Exception as e:
            logger.warning(f"导航失败 {url}: {e}")
            return False
    
    def wait_for_element(
        self,
        locator: tuple,
        timeout: int = 10,
        condition: str = "presence"
    ) -> Any:
        """等待元素
        
        Args:
            locator: 元素定位器 (By.ID, "id")
            timeout: 超时时间
            condition: 等待条件（presence/visibility/clickable）
            
        Returns:
            找到的元素
        """
        try:
            if condition == "presence":
                element = WebDriverWait(self._driver, timeout).until(
                    EC.presence_of_element_located(locator)
                )
            elif condition == "visibility":
                element = WebDriverWait(self._driver, timeout).until(
                    EC.visibility_of_element_located(locator)
                )
            elif condition == "clickable":
                element = WebDriverWait(self._driver, timeout).until(
                    EC.element_to_be_clickable(locator)
                )
            else:
                raise ValueError(f"不支持的等待条件: {condition}")
            
            return element
            
        except TimeoutException:
            logger.warning(f"等待元素超时: {locator}")
            return None
        except Exception as e:
            logger.warning(f"等待元素失败: {e}")
            return None
    
    def execute_script(self, script: str, *args) -> Any:
        """执行JavaScript代码
        
        Args:
            script: JavaScript代码
            *args: 脚本参数
            
        Returns:
            执行结果
        """
        try:
            return self._driver.execute_script(script, *args)
        except Exception as e:
            logger.warning(f"执行脚本失败: {e}")
            return None
    
    def get_page_source(self) -> str:
        """获取页面源码"""
        return self._driver.page_source if self._driver else ""
    
    def get_current_url(self) -> str:
        """获取当前URL"""
        return self._driver.current_url if self._driver else ""
    
    def take_screenshot(
        self,
        output_path: Union[str, Path],
        full_page: bool = False
    ) -> bool:
        """截取页面截图
        
        Args:
            output_path: 输出文件路径
            full_page: 是否截取完整页面
            
        Returns:
            是否成功截图
        """
        if not self._driver:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if full_page:
                # 获取页面完整高度
                total_height = self._driver.execute_script(
                    "return document.body.parentNode.scrollHeight"
                )
                viewport_height = self._driver.execute_script(
                    "return window.innerHeight"
                )
                
                # 设置窗口大小
                original_size = self._driver.get_window_size()
                self._driver.set_window_size(
                    original_size['width'],
                    total_height
                )
                
                # 截图
                success = self._driver.save_screenshot(str(output_path))
                
                # 恢复窗口大小
                self._driver.set_window_size(
                    original_size['width'],
                    original_size['height']
                )
            else:
                success = self._driver.save_screenshot(str(output_path))
            
            return success
            
        except Exception as e:
            logger.warning(f"截图失败: {e}")
            return False 