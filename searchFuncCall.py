# coding: UTF-8
"""
Simple script to search for function calls with a given function name 
in all Python scripts in the current and its sub-folders.

last edited on 2023-06-17
"""

import sys
from os import path, getcwd

from modFFC import *

#-------------------------------------------------------------------------------

def searchStrInFiles(searchStr, searchFileExt=["py"], flagFuncCall=True):
    """ search the given string in files in the current and its sub-folders, 
    returning where (file & line#) it was and a part of the line

    Args:
        searchStr (str): string to search
        searchFileExt (list): list of file types (extensions) to search
        flagFuncCall (bool): string to search is a function call 

    Returns:
        rslt (dict): Result line# and line string 
    """
    rslt = {} 
    fpLst = getFilePaths_recur(getcwd())
    for fp in fpLst:
        fn = path.basename(fp) # file name
        ext = fn.split(".")[-1] # file extension
        if not ext in searchFileExt: continue # ignore this file

        fh = open(fp, "r")
        lines = fh.readlines()
        fh.close()
        _rslt = [] 
        flagInComment = False
        for li, line in enumerate(lines):
       
            if "'''" in line or '"""' in line:
                flagInComment = not flagInComment

            if not searchStr in line: continue
            
            if flagFuncCall:
                if flagInComment: continue # comment
                prcd = line.split(searchStr)[0].strip()
                if "#" in prcd: continue # comment
                if prcd in ["def", "class"]: continue # defining line
            
            ### add the line number and some text
            line = line.strip()
            si = line.index(searchStr)
            aTL = 30
            idx = [max(0, si-aTL), min(si+len(searchStr)+aTL, len(line))]
            _txt = ''
            if idx[0] > 0: _txt += ".. "
            _txt += line[idx[0]:idx[1]]
            if idx[1] < len(line): _txt += " .."
            _rslt.append([li+1, _txt]) # store line# and part of line string

        if _rslt != []: # there was a call in this file
            _rslt.insert(0, path.split(fp)[0]) # store the directory 
            rslt[fn] = _rslt # store the result with the filename as a key
    return rslt

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        funcName = sys.argv[1] # name of the function to search
        rslt = searchStrInFiles(funcName)
        if len(rslt) == 0:
            print(f"No function call of '{funcName}' was found.")
        else:
            for fn in rslt.keys():
                printStr = f"# Filename: {fn} ----------\n"
                _dir = rslt[fn][0] # directory 
                rslt[fn].pop(0)
                printStr += f'{_dir}\n\n'
                printStr += "Line#, Text around the search-string\n\n"
                for ln in rslt[fn]:
                    printStr += f"[{ln[0]}], {ln[1]}\n\n"
                printStr += "\n"
                print(printStr)





