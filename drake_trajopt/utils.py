from functools import cache
from pathlib import Path
from termcolor import cprint, colored
from .timer import Timer, timer

@cache
def GetModelsPath():
    # print('Computing assets path ...')
    return Path(__file__).parent.parent / "models"

def GetPackageXmls():
    return str(GetModelsPath() / 'package.xml')


class Logger:
    @staticmethod
    def INFO(msg, **kwargs):
        cprint(f'INFO: {msg}', color='green', **kwargs)
    
    @staticmethod
    def WARN(msg, **kwargs):
        cprint(f'WARN: {msg}', color='yellow', **kwargs)

    @staticmethod
    def ERROR(msg, raise_exception=True, **kwargs):
        if raise_exception:
            raise Exception(colored(msg, color='red'))
        else:
            cprint(f'ERROR: {msg}', color='red', **kwargs)
