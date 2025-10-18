import json  # noqa
import math  # noqa
import os  # noqa
import re  # noqa
import signal

import matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sympy import Eq, solve, symbols  # noqa


def input(*args, **kwargs):  # noqa
    raise NotImplementedError('Python input() function is disabled.')


def _m6_timout_handler(_signum=None, _frame=None):
    raise TimeoutError('M6_CODE_INTERPRETER_TIMEOUT')


try:
    signal.signal(signal.SIGALRM, _m6_timout_handler)
except AttributeError:  # windows
    pass


class _M6CountdownTimer:

    @classmethod
    def start(cls, timeout: int):
        try:
            signal.alarm(timeout)
        except AttributeError:  # windows
            pass  # I haven't found a timeout solution that works with windows + jupyter yet.

    @classmethod
    def cancel(cls):
        try:
            signal.alarm(0)
        except AttributeError:  # windows
            pass


sns.set_theme()

_m6_font_prop = FontProperties(fname='{{M6_FONT_PATH}}')
plt.rcParams['font.family'] = _m6_font_prop.get_name()
