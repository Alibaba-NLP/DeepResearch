# coding=utf-8
import os
import sys
import logging


import platform


def check_macos():
    os_name = platform.system()
    if os_name == 'Darwin':
        return True
    return False

def setup_logger(level = None, logfile_name: str = 'search-agent'):
    if level is None:
        if os.getenv('SEARCH_AGENT_DEBUG', '0').strip().lower() in ('1', 'true'):
            level = logging.DEBUG
        else:
            level = logging.INFO

    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    # Do not run handler.setLevel(level) so that users can change the level via logger.setLevel later
    handler.setFormatter(formatter)

    # agent path
    if check_macos():
        agent_path = 'logs'
    else:
        agent_path = os.getenv('AGENT_PATH', 'logs')
        if len(agent_path) > 0:
            agent_path = agent_path + '/logs'
    # check path
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
    file_handler = logging.FileHandler(f'{agent_path}/{logfile_name}.log')
    file_handler.setFormatter(formatter)

    _logger = logging.getLogger(logfile_name)
    _logger.setLevel(level)
    _logger.addHandler(handler)
    _logger.addHandler(file_handler)
    return _logger


logger = setup_logger(logfile_name='search-agent')
access_logger = setup_logger(logfile_name='search-agent-access')
error_logger = setup_logger(logfile_name='search-agent-error')
