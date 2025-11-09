# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import functools
import inspect
from typing import Optional
from colorama import Fore, Style, Back


init_loggers = {}

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger(log_file: Optional[str] = None,
               log_level: int = logging.INFO,
               file_mode: str = 'w'):
    """ Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
    """

    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)

    if logger_name in init_loggers:
        add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    init_loggers[logger_name] = True

    return logger


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)


def log_with_location(color=Fore.WHITE):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取调用栈
            stack = inspect.stack()
            # 获取调用此方法的上一层栈帧
            _, filename, line, _, _, _ = stack[1]
            # 获取文件的基本名
            path_parts = filename.split('/')
            relative_path = []
            for part in path_parts[::-1]:
                relative_path.append(part)
                if part == 'evals':
                    break
            filename = '/'.join(relative_path[::-1])
            # 构建带有文件名和行号的消息
            message = f'[{filename}:{line}] {color}{args[0]}' + Style.RESET_ALL
            # 调用原始日志方法
            func(message, *args[1:], **kwargs)

        return wrapper

    return decorator


class Logger:
    logger = get_logger()

    @staticmethod
    @log_with_location(Fore.WHITE)
    def debug(msg):
        Logger.logger.debug(str(msg))

    @staticmethod
    @log_with_location(Fore.WHITE)
    def info(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.YELLOW)
    def warning(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.RED)
    def error(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Back.RED)
    def critical(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.WHITE)
    def white(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.GREEN)
    def green(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.RED)
    def red(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.BLUE)
    def blue(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.BLACK)
    def black(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Back.WHITE)
    def bwhite(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Back.GREEN)
    def bgreen(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Back.RED)
    def bred(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Back.BLUE)
    def bblue(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Back.BLACK)
    def bblack(msg):
        Logger.logger.info(str(msg))

    @staticmethod
    @log_with_location(Fore.CYAN)
    def cyan(msg):
        Logger.logger.info(str(msg))
