#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor项目安装配置文件

该文件定义了WebSailor包的安装配置，包括项目元数据、依赖关系和入口点。
WebSailor是一个专注于超人推理能力的Web智能体框架。

作者: Evan Zuo
日期: 2025年1月
"""

from setuptools import setup, find_packages
import os

# 读取README文件作为长描述
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements.txt文件
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

setup(
    name="websailor",
    version="0.1.0",
    author="Evan Zuo",
    author_email="websailor@example.com",
    description="WebSailor: 具备超人推理能力的Web智能体框架",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/websailor/websailor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "websailor=websailor.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 