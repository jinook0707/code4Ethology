# coding: UTF-8
""" for init. some global variables including a logger for an app

last edited: 2023-11-08
"""

import sys, logging
from os import path

#-------------------------------------------------------------------------------

def setMyLogger(name, formatStr=""):
    """ Set logger

    Args:
        name (str): Name of the logger
        formatStr (str): Format string

    Returns:
        (logging.Logger): Logger
    """
    myLogger = logging.getLogger(name)
    myLogger.setLevel(logging.DEBUG)
    LSH = logging.StreamHandler()
    LSH.setLevel(logging.DEBUG)
    if formatStr == "":
        formatStr = "%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s"
    LSH.setFormatter(logging.Formatter(formatStr))
    myLogger.addHandler(LSH)
    return myLogger

#-------------------------------------------------------------------------------

def init(pythonFile):
    global FPATH, P_DIR, MyLogger

    rPath = path.realpath(pythonFile)
    FPATH, fn = path.split(rPath) # path of where the Python file is & filename 
    sys.path.append(FPATH) # add FPATH to path
    
    _path = path.realpath(__file__)
    P_DIR = path.split(_path)[0] # path of where this initApp file is 

    #from modFFC import setMyLogger
    MyLogger = setMyLogger(fn.split(".")[0])

#-------------------------------------------------------------------------------


