import logging
import os


def setup_logger(level=None):
    if level is None:
        if os.getenv('QWEN_AGENT_DEBUG', '0').strip().lower() in ('1', 'true'):
            level = logging.DEBUG
        else:
            level = logging.INFO

    handler = logging.StreamHandler()
    # Do not run handler.setLevel(level) so that users can change the level via logger.setLevel later
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    _logger = logging.getLogger('qwen_agent_logger')
    _logger.setLevel(level)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()
