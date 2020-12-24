import os


def projectpath():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def pdsettings():
    import pandas as pd
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
