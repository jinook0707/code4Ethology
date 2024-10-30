# coding: UTF-8
"""
Frequenty used functions and classes

Dependency:
    wxPython (4.0), 
    Numpy (1.17), 
    SciPy (1.4), 

last edited on 2024-08-20
"""

import sys, errno, colorsys, re, traceback, pickle, pathlib
from threading import Thread
from os import path, strerror, system
from time import time, sleep
from datetime import datetime
from glob import glob
from copy import copy

import wx, wx.grid, wx.stc 
import wx.lib.scrolledpanel as SPanel
import wx.lib.agw.gradientbutton as gBtn
from wx.adv import Animation, AnimationCtrl
import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt

from initVars import *

if not "ICON_FP" in globals():
    _path = path.realpath(__file__)
    ICON_FP = path.join(path.split(_path)[0], "image", "icon.png")

DEBUG = False

#-------------------------------------------------------------------------------

def GNU_notice(idx=0):
    """ Function for printing GNU copyright statements

    Args:
        idx (int): Index to determine which statement to print out.

    Returns:
        None

    Examples:
        >>> GNU_notice(0)
        Copyright (c) ...
        ...
        run this program with option '-c' for details.
    """
    if DEBUG: MyLogger.info(str(locals()))

    if idx == 0:
        year = datetime.now().year
        msg = "Copyright (c) %i Jinook Oh, Sylvia Cremer.\n"%(year)
        msg += "This program comes with ABSOLUTELY NO WARRANTY;"
        msg += " for details run this program with the option `-w'."
        msg += "This is free software, and you are welcome to redistribute"
        msg += " it under certain conditions;"
        msg += " run this program with the option `-c' for details."
    elif idx == 1:
        msg = "THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED"
        msg += " BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING"
        msg += " THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE"
        msg += " PROGRAM 'AS IS' WITHOUT WARRANTY OF ANY KIND, EITHER"
        msg += " EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE"
        msg += " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A"
        msg += " PARTICULAR PURPOSE."
        msg += " THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE"
        msg += " PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU"
        msg += " ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR"
        msg += " CORRECTION."
    elif idx == 2:
        msg = "You can redistribute this program and/or modify it under" 
        msg += " the terms of the GNU General Public License as published"
        msg += " by the Free Software Foundation, either version 3 of the"
        msg += " License, or (at your option) any later version."
    print(msg)

#-------------------------------------------------------------------------------

def chkFPath(fp):
    """ Check whether file/folder exists
    If not found, raise FileNotFoundError

    Args:
        fp: file or folder path to check

    Returns:
        None
    
    Examples:
        >>> chkFPath('./test/test1.txt')

    Raises:
       FileNotFoundError: When 'fp' is not a valid file-path. 
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    rslt = False 
    if path.isdir(fp): rslt = True
    elif path.isfile(fp): rslt = True
    if rslt == False:
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), fp)

#-------------------------------------------------------------------------------

def get_time_stamp(flag_ms=False):
    """ Function to return string which contains timestamp.

    Args:
        flag_ms (bool, optional): Whether to return microsecond or not

    Returns:
        ts_ret (str): Timestamp string to return

    Examples:
        >>> print(get_time_stamp())
        2019_09_10_16_21_56
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    ts = datetime.now()
    ts_ret = ('%.4i_%.2i_%.2i_%.2i_%.2i_%.2i')%(ts.year, 
                                                ts.month, 
                                                ts.day, 
                                                ts.hour, 
                                                ts.minute, 
                                                ts.second)
    if flag_ms == True: ts_ret += '_%.6i'%(ts.microsecond)
    return ts_ret

#-------------------------------------------------------------------------------

def get_datetime(tsStr, flagMicroSec=False):
    """ Function to return datetime made from given string of timestamp.

    Args:
        tsStr (str): Timestamp string (from get_time_stamp)
        flagMicroSec (bool): Whether to include microseconds

    Returns:
        tsDT (datetime.datetime): Datetime object from tsStr. 

    Examples:
        >>> print(get_datetime("2019_09_10_16_21_56"))
        datetime.datetime(2019, 9, 10, 16, 21, 56)
    """
    if DEBUG: MyLogger.info(str(locals()))

    if "_" in tsStr:
    # timestamp from get_time_stamp(); e.g.: '2023_11_28_23_59_01'
        tsLst = [int(x) for x in tsStr.split("_")]
    else: 
    # assume timestamp string such as '2023-11-28 23:59:01'
    #   of '2023-11-28T23:59:01'
        if "T" in tsStr: splitChr = "T"
        else: splitChr = " "
        _tsL = tsStr.split(splitChr)
        if len(_tsL) == 0 or len(_tsL) > 2: return False
        tsLst = []
        splitChr = ["-", ":"]
        for i in range(len(_tsL)):
            for _ts in _tsL[i].split(splitChr[i]):
                if "." in _ts:
                    sec, mSec = _ts.split(".")
                    tsLst += [int(sec), int(mSec)]
                else:
                    tsLst.append(int(_ts))

    if len(tsLst) < 3: return False
    tsDT = datetime(year=tsLst[0], month=tsLst[1], day=tsLst[2])
    for i in range(3, len(tsLst)):
        if i == 3: tsDT = tsDT.replace(hour=tsLst[i])
        elif i == 4: tsDT = tsDT.replace(minute=tsLst[i])
        elif i == 5: tsDT = tsDT.replace(second=tsLst[i])
        elif i == 6 and flagMicroSec: tsDT = tsDT.replace(microsecond=tsLst[i])
    return tsDT

#-------------------------------------------------------------------------------

def getFileMTime(fp):
    """ Function to return a file's last modification time 

    Args:
        fp (str): Input filepath.

    Returns:
        (datetime): Datetime to return.

    Examples:
        >>> getFileMTime('test.mp4') 
        datetime.datetime(2022, 2, 8, 17, 19, 3)
    """
    if DEBUG: MyLogger.info(str(locals()))

    if not path.isfile(fp): return
    f = pathlib.Path(fp)
    return datetime.fromtimestamp(f.stat().st_mtime)

#-------------------------------------------------------------------------------

def writeFile(file_path, txt='', mode='a'):
    """ Function to write a text or numpy file.

    Args:
        file_path (str): File path for output file.
        txt (str): Text to print in the file.
        mode (str, optional): File opening mode.

    Returns:
        None

    Examples:
        >>> writeFile('logFile.txt', 'A log is written.', 'a')
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    f = open(file_path, mode)
    f.write(txt)
    f.close()

#-------------------------------------------------------------------------------

def getFilePaths_recur(folderPath, flagIncludeDir=False):
    """ Returns paths of all files under the given path (folderPath), 
    finding files recursively in all sub-folders of the folderPath. 

    Args:
        folderPath (str): Folder path to start.
        flagIncludDir (bool): Whether to include folder-names.

    Returns:
        fpLst (list): List of the found file paths. 
    """
    if DEBUG: MyLogger.info(str(locals()))

    fpLst = []
    for fp in sorted(glob(path.join(folderPath, "*"))):
        if path.isdir(fp):
            if flagIncludeDir: fpLst.append(fp)
            fpLst += getFilePaths_recur(fp, flagIncludeDir)
        elif path.isfile(fp):
            fpLst.append(fp)

    return fpLst

#-------------------------------------------------------------------------------

def str2num(s, c=''):
    """ Function to convert string to an integer or a float number.
    
    Args: 
        s (str): String to process 
        c (str): Intented conversion

    Returns:
        oNum (None/ int/ float):
          Converted number or None (when it failed to convert).

    Examples:
        >>> print(str2num('test'))
        None
        >>> print(str2num('3'))
        3 
        >>> print(str2num('3.0'))
        3.0
        >>> print(str2num('3.0', 'int'))
        3
    """
    if DEBUG: MyLogger.info(str(locals()))
   
    if s == '':
        return 0
    
    else:
        oNum = None 
        if c != '': # conversion method is given
            try: oNum = eval('%s(%s)'%(c, s)) # try the intended conversion
            except: pass
        else: # no conversion is specified
            try:
                oNum = int(s) # try to convert to integer first
            except:
                try: oNum = float(s) # then, float
                except: pass
        return oNum 

#-------------------------------------------------------------------------------

def natural_keys(txt):
    """ Key function for Python's sort function.
    For sorting string including numbers in it.

    Args:
        txt (str): Input string.

    Returns:
        (list): Sorted list.

    Examples:
        >>> inputList = ['item1', 'item10', 'item2']
        >>> inputList.sort(key=natural_keys)
        >>> print(inputList)
        ['item1', 'item2', 'item10']
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    def atoi(txt):
        return int(txt) if txt.isdigit() else txt
    return [atoi(c) for c in re.split('(\d+)',txt)]

#-------------------------------------------------------------------------------

def lst2rng(inputLst):
    """ Convert a list to a list of ranges; change consecutive numbers into
    a range ([beginning-number, end-number]).
    
    Args: 
        inputLst (list): list of integers 

    Returns:
        outputLst (list): list of ranges

    Examples:
        >>> lst2rng([1, 2, 3, 5, 6, 10, 11, 12, 100, 102, 103]) 
        [[1, 3], [5, 6], [10, 12], [100, 100], [102, 103]]
    """
    if DEBUG: MyLogger.info(str(locals()))

    outputLst = [[inputLst[0]]]
    consecIdx = outputLst[-1][0]
    for i in range(1, len(inputLst)):
        currIdx = inputLst[i]
        consecIdx += 1
        if consecIdx == currIdx:
            continue
        else:
            outputLst[-1].append(consecIdx-1)
            outputLst.append([currIdx]) 
            consecIdx = outputLst[-1][0]
    if len(outputLst[-1]) == 1:
        outputLst[-1].append(consecIdx)
    return outputLst

#-------------------------------------------------------------------------------

def load_img(fp, size=(-1,-1)):
    """ Load an image using wxPython.

    Args:
        fp (str): File path of an image to load. 
        size (tuple): Output image size.

    Returns:
        img (wx.Image)

    Examples:
        >>> img1 = load_img("test.png")
        >>> img2 = load_img("test.png", size=(300,300))
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    chkFPath(fp) # chkeck whether file exists
    
    tmp_null_log = wx.LogNull() # for not displaying 
      # the tif library warning
    img = wx.Image(fp, wx.BITMAP_TYPE_ANY)
    del tmp_null_log
    if size != (-1,-1) and type(size[0]) == int and \
      type(size[1]) == int: # appropriate size is given
        if img.GetSize() != size:
            img = img.Rescale(size[0], size[1], wx.IMAGE_QUALITY_HIGH)
    
    return img

#-------------------------------------------------------------------------------
    
def drawBMPonMemDC(bmpSz, dFunc):
    """ Make memoryDC and draw bitmap with given drawing function 
    
    Args:
        bmpSz (tuple): Width and height of bitmap to draw
        dFunc (function): To draw something on bitmap 
    
    Returns:
        bmp (wx.Bitmap): Drawn bitmap image 
    """
    if DEBUG: MyLogger.info(str(locals()))

    bmp = wx.Bitmap(bmpSz[0], bmpSz[1], depth=32)
    memDC = wx.MemoryDC()
    memDC.SelectObject(bmp)
    memDC.SetBackground(wx.Brush((0,0,0,255)))
    memDC.Clear()
    dFunc(memDC) # draw 
    memDC.SelectObject(wx.NullBitmap)
    return bmp 

#-------------------------------------------------------------------------------

def set_img_for_btn(imgPath, btn, imgPCurr=None, imgPDis=None, 
                    imgPFocus=None, imgPPressed=None):
    """ Set image(s) for a wx.Button

    Args:
        imgPath (str): Path of default image file. 
        btn (wx.Button): Button to put image(s).
        imgPCurr (str): Path of image for when mouse is over.
        imgPDis (str): Path of image for when button is disabled.
        imgPFocus (str): Path of image for when button has the keyboard focus.
        imgPPressed (str): Path of image for when button was pressed.

    Returns:
        btn (wx.Button): Button after processing.

    Examples:
        >>> btn=set_img_for_btn('btn1img.png',wx.Button(self, -1,'testButton'))
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    imgPaths = dict(all=imgPath, current=imgPCurr, disabled=imgPDis,
                    focus=imgPFocus, pressed=imgPPressed)
    for key in imgPaths.keys():
        fp = imgPaths[key]
        if fp == None: continue
        img = load_img(fp)
        bmp = wx.Bitmap(img)
        if key == 'all': btn.SetBitmap(bmp)
        elif key == 'current': btn.SetBitmapCurrent(bmp)
        elif key == 'disabled': btn.SetBitmapDisabled(bmp)
        elif key == 'focus': btn.SetBitmapFocus(bmp)
        elif key == 'pressed': btn.SetBitmapPressed(bmp)
    return btn

#-------------------------------------------------------------------------------

def getWXFonts(initFontSz=8, numFonts=5, fSzInc=2, 
               fontFaceName="Arial", weight=wx.FONTWEIGHT_NORMAL, 
               style=wx.FONTSTYLE_NORMAL, underline=False):
    """ For setting up several fonts (wx.Font) with increasing size.

    Args:
        initFontSz (int): Initial (the smallest) font size.
        numFonts (int): Number of fonts to return.
        fSzInc (int): Increment of font size.
        fontFaceName (str, optional): Font face name.
        weight (int): Font weight.
        style (int): Font style.
        underline (bool): To underline text or not.

    Returns:
        fonts (list): List of several fonts (wx.Font)

    Examples:
        >>> fonts = getWXFonts(8, 3)
        >>> fonts = getWXFonts(8, 3, 5, 'Arial')
    """
    if DEBUG: MyLogger.info(str(locals()))

    if fontFaceName == "":
        if 'darwin' in sys.platform: fontFaceName = "Monaco"
        else: fontFaceName = "Courier"
    fontSz = initFontSz
    fonts = []  # larger fonts as index gets larger 
    for i in range(numFonts):
        fonts.append(
                        wx.Font(
                                fontSz, 
                                wx.FONTFAMILY_SWISS, 
                                style, 
                                weight,
                                underline, 
                                faceName=fontFaceName,
                               )
                    )
        fontSz += fSzInc 
    return fonts

#-------------------------------------------------------------------------------

def setupPanel(w, wxFrame, pk, useSPanel=True):
    """ Set up wxPython (scrolled) panel

    Args:
        w (list): List of widget data
        wxFrame (wx.Frame): Frame to work on
        pk (str): Panel key string
        useSPanel (bool): whether to use ScrolledPanel. If False, use wx.Panel

    Return:
        widLst (list): List of made widgets 
        pSz (list): Resultant panel size after adding widgets
    """
    if DEBUG: MyLogger.info(str(locals()))

    pi = wxFrame.pi[pk]

    if not pk in wxFrame.panel.keys():
    # panel is not created yet
        if useSPanel: p = SPanel.ScrolledPanel
        else: p = wx.Panel
        panel = p(wxFrame, pos=pi["pos"], size=pi["sz"], style=pi["style"])
        panel.SetBackgroundColour(pi["bgCol"])
        panel.panelKey = pk # store panel key string
        panel.mousePressedPt = [-1, -1]
        wxFrame.panel[pk] = panel 
        wxFrame.gbs[pk] = wx.GridBagSizer(0,0)
    else:
        panel = wxFrame.panel[pk]

    # set up wxPython widgets
    widLst, pSz = addWxWidgets(w, wxFrame, pk)

    # resize panel, if necessary
    if pSz[0] > pi["sz"][0]: panel.SetSize(pSz[0], pi["sz"][1])

    wxFrame.panel[pk].SetSizer(wxFrame.gbs[pk])
    wxFrame.gbs[pk].Layout()
    if useSPanel: wxFrame.panel[pk].SetupScrolling()
    wxFrame.panel[pk].SetDoubleBuffered(True) # for counteracting flicker
   
    if hasattr(wxFrame, "onKeyPress") and \
      callable(getattr(wxFrame, "onKeyPress")):
        wxFrame.panel[pk].Bind(wx.EVT_CHAR_HOOK, wxFrame.onKeyPress)  

    return widLst

#-------------------------------------------------------------------------------

def addWxWidgets(w, self, pk):
    """ Set up wxPython widgets

    Args:
        w (list): List of widget data
        self (wx.Frame): Frame to work on
        pk (str): Panel key string

    Return:
        widLst (list): List of made widgets 
        pSz (list): Resultant panel size after adding widgets
    """
    if DEBUG: MyLogger.info(str(locals()))

    panel = self.panel[pk]
    gbs = self.gbs[pk]

    pSz = [0, 0] # panel size
    widLst = []
    row = 0
    for ri in range(len(w)):
        _width = 0
        _height = 0
        col = 0
        for ci in range(len(w[ri])):
            wd = w[ri][ci]
            if "size" in wd.keys(): size = wd["size"]
            else: size = (-1, -1)
            if "style" in wd.keys(): style = wd["style"]
            else: style = 0
            if "id" in wd.keys(): wId = wd["id"]
            else: wId = -1
            if wd["type"] == "sTxt": # wx.StaticText
                _w = wx.StaticText(panel, wId, label=wd["label"], size=size,
                                   style=style) 

            elif wd["type"] == "sBmp": # wx.StaticBitmap
                if "bmp" in wd.keys(): bmp = wd["bmp"]
                else: bmp = wx.NullBitmap
                _w = wx.StaticBitmap(panel, wId, size=size, bitmap=bmp)

            elif wd["type"] == "sLn": # wx.StaticLine
                _w = wx.StaticLine(panel, wId, size=size, style=style) 
            
            elif wd["type"] == "txt": # wx.TextCtrl
                _w = wx.TextCtrl(panel, wId, value=wd["val"], size=size, 
                                 style=style)
                if "numOnly" in wd.keys() and wd["numOnly"]:
                    _w.Bind(
                        wx.EVT_CHAR, 
                        lambda event: self.onTextCtrlChar(event,
                                                          isNumOnly=True)
                        )
                if "procEnter" in wd.keys() and wd["procEnter"]:
                    if hasattr(self, "onEnterInTextCtrl") and \
                      callable(getattr(self, "onEnterInTextCtrl")): 
                        _w.Bind(wx.EVT_TEXT_ENTER, self.onEnterInTextCtrl)
            
            elif wd["type"] == "btn": # wx.Button
                lbl = ""
                if "label" in wd.keys(): lbl = wd["label"]
                _w = wx.Button(panel, wId, label=lbl, size=size,
                               style=style)
                if "img" in wd.keys(): set_img_for_btn(wd["img"], _w) 
                if hasattr(self, "onButtonPressDown") and \
                  callable(getattr(self, "onButtonPressDown")): 
                    _w.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)

            elif wd["type"] == "gBtn":
            # wx.lib.agw.gradientbutton.GradientButton
                lbl = ""
                if "label" in wd.keys(): lbl = wd["label"]
                _w = gBtn.GradientButton(panel, wId, label=lbl, size=size,
                                         style=style)
                if "img" in wd.keys(): set_img_for_btn(wd["img"], _w)
                bgCol = (0,0,0) 
                fgCol = (255,255,255) 
                if "bgColor" in wd.keys(): bgCol = wx.Colour(wd["bgColor"])
                if "fgColor" in wd.keys(): fgCol = wx.Colour(wd["fgColor"])
                _w.SetBaseColours(bgCol, fgCol)
                if "topSCol" in wd.keys(): topSCol = wx.Colour(wd["topSCol"])
                else: topSCol = bgCol
                if "topECol" in wd.keys(): topECol = wx.Colour(wd["topECol"])
                else: topECol = bgCol
                _w.SetTopStartColour(topSCol)
                _w.SetTopEndColour(topECol)
                if "botSCol" in wd.keys(): botSCol = wx.Colour(wd["botSCol"])
                else: botSCol = bgCol
                if "botECol" in wd.keys(): botECol = wx.Colour(wd["botECol"])
                else: botECol = bgCol
                _w.SetBottomStartColour(botSCol)
                _w.SetBottomEndColour(botECol)
                if hasattr(self, "onButtonPressDown") and \
                  callable(getattr(self, "onButtonPressDown")): 
                    _w.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
            
            elif wd["type"] == "chk": # wx.CheckBox
                _w = wx.CheckBox(panel, id=wId, label=wd["label"], size=size,
                                 style=style)
                val = False
                if "val" in wd.keys(): val = wd["val"]
                _w.SetValue(val)
                if hasattr(self, "onCheckBox") and \
                  callable(getattr(self, "onCheckBox")): 
                    _w.Bind(wx.EVT_CHECKBOX, self.onCheckBox)
            
            elif wd["type"] == "cho": # wx.Choice
                _w = wx.Choice(panel, wId, choices=wd["choices"], size=size,
                               style=style)
                if hasattr(self, "onChoice") and \
                  callable(getattr(self, "onChoice")): 
                    _w.Bind(wx.EVT_CHOICE, self.onChoice)
                if "val" in wd.keys():
                    _w.SetSelection(_w.FindString(wd["val"]))
            
            elif wd["type"] == "radB": # wx.RadioBox
                _w = wx.RadioBox(panel, wId, label=wd["label"], size=size,
                                 choices=wd["choices"], style=style,
                                 majorDimension=wd["majorDimension"])
                if hasattr(self, "onRadioBox") and \
                  callable(getattr(self, "onRadioBox")): 
                    _w.Bind(wx.EVT_RADIOBOX, self.onRadioBox)
                _w.SetSelection(_w.FindString(wd["val"]))
            
            elif wd["type"] == "sld": # wx.Slider
                _w = wx.Slider(panel, wId, size=size, value=wd["val"],
                               minValue=wd["minValue"], maxValue=wd["maxValue"],
                               style=style)
                if hasattr(self, "onSlider") and \
                  callable(getattr(self, "onSlider")): 
                    _w.Bind(wx.EVT_SCROLL, self.onSlider)
            
            elif wd["type"] == "spin": # wx.SpinCtrl
                if "double" in wd.keys() and wd["double"]:
                    _w = wx.SpinCtrlDouble(panel, wId, size=size,
                                           min=wd["min"], max=wd["max"],
                                           inc=wd["inc"], initial=wd["initial"],
                                           style=style)
                else:
                    _w = wx.SpinCtrl(panel, wId, size=size, min=wd["min"],
                                     max=wd["max"], initial=wd["initial"],
                                     style=style)
                if hasattr(self, "onSpinCtrl") and \
                  callable(getattr(self, "onSpinCtrl")): 
                    _w.Bind(wx.EVT_SPINCTRL, self.onSpinCtrl)
                if hasattr(self, "onEnterInTextCtrl") and \
                  callable(getattr(self, "onEnterInTextCtrl")):
                    _w.Bind(wx.EVT_TEXT_ENTER, self.onEnterInTextCtrl)
            
            elif wd["type"] == "cPk": # wx.ColourPickerCtrl
                _w = wx.ColourPickerCtrl(panel, wId, size=size, 
                                         colour=wd["color"], style=style)
                if hasattr(self, "onColourPicker") and \
                  callable(getattr(self, "onColourPicker")): 
                    _w.Bind(wx.EVT_COLOURPICKER_CHANGED, self.onColourPicker)
            
            elif wd["type"] == "grid": # wx.grid.Grid
                _w = Grid(panel, np.empty((0,0), dtype=np.uint8))
                if hasattr(self, "onDataGridCellChanged") and \
                  callable(getattr(self, "onDataGridCellChanged")): 
                    self.Bind(wx.grid.EVT_GRID_CELL_CHANGED, 
                              self.onDataGridCellChanged)
                if hasattr(self, "onDataGridCellSelected") and \
                  callable(getattr(self, "onDataGridCellSelected")): 
                    self.Bind(wx.grid.EVT_GRID_SELECT_CELL, 
                              self.onDataGridCellSelected)
                if hasattr(self, "onDataGridCellsSelected") and \
                  callable(getattr(self, "onDataGridCellsSelected")): 
                    self.Bind(wx.grid.EVT_GRID_RANGE_SELECT, 
                              self.onDataGridCellsSelected)
            
            elif wd["type"] == "stc": # wx.stc.StyledTextCtrl
                _w = STC(panel, pos=(0,0), size=size)

            elif wd["type"] == "panel": # wx.Panel
                _w = wx.Panel(panel, wId, size=size, style=style)
            
            if "name" in wd.keys(): _w.SetName("%s_%s"%(wd["name"], wd["type"]))
            if "wrapWidth" in wd.keys(): _w.Wrap(wd["wrapWidth"])
            if "font" in wd.keys(): _w.SetFont(wd["font"])
            if "fgColor" in wd.keys(): _w.SetForegroundColour(wd["fgColor"]) 
            if "bgColor" in wd.keys(): _w.SetBackgroundColour(wd["bgColor"])
            if "tooltip" in wd.keys(): _w.SetToolTip(wd["tooltip"])
            if "disable" in wd.keys() and wd["disable"]: _w.Disable()
            widLst.append(_w)
            if "border" in wd.keys(): bw = wd["border"]
            else: bw = 5 
            if "flag" in wd.keys(): flag = wd["flag"]
            else: flag = (wx.ALIGN_CENTER_VERTICAL|wx.ALL)
            
            add2gbs(gbs, _w, (row,col), (1, wd["nCol"]), bw=bw, flag=flag)
            _width += _w.GetSize()[0]
            if _w.GetSize()[1] > _height: _height = _w.GetSize()[1]
            col += wd["nCol"] 
        row += 1
        if _width > pSz[0]: pSz[0] = _width
        pSz[1] += _height + 10 

    return widLst, pSz

#-------------------------------------------------------------------------------

def updateFrameSize(wxFrame, w_sz):
    """ Set window size exactly to a user-defined window size (w_sz)
    , excluding counting menubar/border/etc.

    Args:
        wxFrame (wx.Frame): Frame to resize.
        w_sz (tuple): Client size. 

    Returns:
        None

    Examples:
        >>> updateFrameSize(self, (800,600))
    """
    if DEBUG: MyLogger.info(str(locals()))

    ### set window size to w_sz, excluding counting menubar/border/etc.
    _diff = (wxFrame.GetSize()[0]-wxFrame.GetClientSize()[0], 
             wxFrame.GetSize()[1]-wxFrame.GetClientSize()[1])
    _sz = (w_sz[0]+_diff[0], w_sz[1]+_diff[1])
    wxFrame.SetSize(_sz) 
    wxFrame.Refresh()

#-------------------------------------------------------------------------------

def calcI2DRatio(iSz, dispSz, flagEnlarge=False, frac=0.95): 
    """ Calculate ratio for resizing frame image to fit
        the display (in StaticBitmap, paintDC, etc)

    Args:
        iSz (tuple): Image size
        dispSz (tuple): Width and height of StaticBitmap to display image.
        flagEnlarge (bool): If True, also calculate ratio when img size is
                            smaller than display size.
        frac (float): Fraction of the panel size to use.

    Returns:
        r (float): Float number for resizing image later.
    """ 
    if DEBUG: MyLogger.info(str(locals()))

    if flagEnlarge or (iSz[0] > dispSz[0] or iSz[1] > dispSz[1]):
        r1 = float(dispSz[0]*frac) / iSz[0]
        r2 = float(dispSz[1]*frac) / iSz[1]
        r = min(r1, r2)
    else:
        r = 1.0
    return r 

#-------------------------------------------------------------------------------

def configuration(wxFrame, flag, configDefV, panel):
    """ saving/loading configuration (values of certain widgets) of the app 

    Args:
        wxFrame (wx.Frame)
        flag (str): save or load
        configDefV (dict): Default values of config to load.
        panel (wx.Panel/None): Panel to find widgets when saving.

    Returns:
        None
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    configW = wxFrame.configW # widget name list for configuration
    configFP = wxFrame.configFP # file path of configuration file

    if flag == "save":
        config = {}
        for wn in configW:
            w = wx.FindWindowByName(wn, panel)
            config[wn] = widgetValue(w) # get the widget value 
        if hasattr(wxFrame, "viewMenuItem"):
            for vmk in wxFrame.viewMenuItem.keys():
                if wxFrame.viewMenuItem[vmk].IsChecked():
                    config["view_%s"%(vmk)] =  True
                else:
                    config["view_%s"%(vmk)] = False
        fh = open(configFP, "wb")
        pickle.dump(config, fh)
        fh.close()
        return

    elif flag == "load": 
        if path.isfile(configFP):
        # config file exists
            fh = open(configFP, "rb")
            config = pickle.load(fh)
            fh.close()
            for wn in configW:
                if not wn in config.keys():
                    config[wn] = configDefV[wn] # get default value
            if hasattr(wxFrame, "viewMenuItem"):
                for vmk in wxFrame.viewMenuItem.keys():
                    k = "view_%s"%(vmk)
                    if not k in config.keys():
                        config[k] = configDefV[k] 
        else:
        # no config file found
            config = configDefV # use the default values
        return config

#-------------------------------------------------------------------------------

def add2gbs(gbs, 
            widget, 
            pos, 
            span=(1,1), 
            bw=5, 
            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL):
    """ Add 'widget' to given 'gbs'.
    
    Args:
        gbs (wx.GridBagSizer).
        widget (wxPython's widget such as wx.StaticText, wx.Choice, ...).
        pos (tuple): x and y cell indices for positioning 'widget' in 'gbs'.
        span (tuple): width and height in terms of cells in 'gbs'.
        bw (int): Border width.
        flag (int): Flags for styles.
    
    Returns:
        None
    
    Examples:
        >>> add2gbs(self.gbs["ui"], sTxt, (0,0), (1,1))
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    gbs.Add(widget, pos=pos, span=span, border=bw, flag=flag)

#-------------------------------------------------------------------------------

def widgetValue(w, val2set="", flag="get", where2find=None):
    """ Get/Set the value of the given wxPython widget

    Args:
        w (wxPython widget/ str)
        val2set (str): Value to set for the given widget
        flag (str): get or set
        where2find (None/wx.Frame/wx.Panel): Where to find the widget,
            when 'w' is a string.

    Returns:
        rVal (str/None): return value of the widget, None when flag == "set"
    """
    if DEBUG: MyLogger.info(str(locals()))

    if type(w) == str:
        w = wx.FindWindowByName(w, where2find)

    wT1 = [wx.TextCtrl, wx.SpinCtrl, wx.SpinCtrlDouble, wx.CheckBox, wx.Slider]
    wT2 = [wx.Choice, wx.RadioBox]
    rVal = "" 
    if type(w) in wT1:
        if flag == "get":
            rVal = w.GetValue()
        elif flag == "set":
            w.SetValue(val2set)
            rVal = None
    elif type(w) in wT2:
        if flag == "get":
            rVal = w.GetString(w.GetSelection()) # text of the chosen option
        elif flag == "set":
            w.SetSelection(w.FindString(val2set))
            rVal = None
    return rVal

#-------------------------------------------------------------------------------

def preProcUIEvt(frame, event, objName, objType=""):
    """ Conduct some common processes when there was UI event in
    an wxPython object.
    
    Args:
        frame (wx.Frame): Frame that calls this function 
        event (wx.Event)
        objName (str): Name of the widget, supposed to cause the event.
        objType (str): [!! Currently NOT Used !!] Type of the widget
    
    Returns:
        flag_term (bool): flag to quit function
        obj (object): widget that caused the event.
        objName (str): Name of the object.
        wasFuncCalledViaWxEvent (bool): whehter caller function was 
            called via wx.Event or directly called from other function.
        objVal (str): Value of the object, if available.
    """
    if DEBUG: MyLogger.info(str(locals()))

    if objName == "":
        obj = event.GetEventObject()
        objName = obj.GetName()
        wasFuncCalledViaWxEvent = True
    else:
    # funcion was called by some other function without wx.Event
        obj = wx.FindWindowByName(objName, frame)
        wasFuncCalledViaWxEvent = False
   
    if frame.flags["blockUI"] or obj.IsEnabled() == False: flag_term = True 
    else: flag_term = False

    objVal = widgetValue(obj) # get the widget value
    
    return flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal 

#-------------------------------------------------------------------------------
    
def startTaskThread(wxFrame, taskName, targetFunc, args=None,
                    flagBlockUI=True, bgCol=(0,0,10), wmSz="full"): 
    """ start a thread and timer to deal with the given task

    Args:
        wxFrame (wx.wxFrame): wxFrame that calls this function 
        taskName (str): Name of the task.
        targetFunc (func): Function to run for the task
        args (tuple): Arguments to pass to the target function
        flagBlockUI (bool): blocking UI flag in the wxFrame
        bgCol (tuple): Background color of waiting message panel.
        wmSz (str): Size of waiting message panel.

    Returns:
        None
    """
    if DEBUG: MyLogger.info(str(locals()))

    # block user input until the task is finished
    wxFrame.flags["blockUI"] = flagBlockUI 

    if wmSz == "full":
        # hide panels of the frame
        for pk in wxFrame.panel.keys(): wxFrame.panel[pk].Hide()

    # show waiting message panel
    wx.CallLater(1, makeWaitingMessagePanel, wxFrame, wxFrame.wSz,
                 msg="Please wait...", imgFP=ICON_FP, bgCol=bgCol, wmSz=wmSz)
        
    ### set timer for processing messages and data 
    wxFrame.timer[taskName] = wx.Timer(wxFrame)
    wxFrame.Bind(wx.EVT_TIMER,
              lambda event: wxFrame.onTimer(event, taskName),
              wxFrame.timer[taskName])
    wxFrame.timer[taskName].Start(10)
   
    ### start thread to write
    if args == None: wxFrame.th = Thread(target=targetFunc)
    else: wxFrame.th= Thread(target=targetFunc, args=args)
    wx.CallLater(100, wxFrame.th.start)

#-------------------------------------------------------------------------------
    
def postProcTaskThread(wxFrame, flag):
    """ common processing after finishing a task thread 

    Args:
        wxFrame (wx.wxFrame): wxFrame that calls this function 
        flag (str): flag of the task thread 

    Returns:
        None
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    wxFrame.flags["blockUI"] = False
    if wxFrame.th != None:
        try: wxFrame.th.join()
        except: pass
    if flag in wxFrame.timer.keys():
        try: wxFrame.timer[flag].Stop()
        except: pass
    try: wxFrame.tmpWaitingMsgPanel.Destroy() # destroy tmp. panel
    except: pass
    for pk in wxFrame.panel.keys(): wxFrame.panel[pk].Show() # show other panels
    if "tp" in wxFrame.panel.keys(): wxFrame.panel["tp"].SetFocus()

#-------------------------------------------------------------------------------

def stopAllTimers(timer):
    """ Stop all running wxPython timers
    
    Args:
        timer (dict): container of all timers to stop.
    
    Returns:
        timer (dict) 
    """
    if DEBUG: MyLogger.info(str(locals()))

    for k in timer.keys():
        if timer[k] != None:
            try: timer[k].Stop()
            except: pass
            timer[k] = None
    return timer

#-------------------------------------------------------------------------------

def makeWaitingMessagePanel(wxFrame, wSz, msg="Please wait...", 
                            imgFP=ICON_FP, bgCol=(75,75,75), wmSz="full"):
    """ Make a panel showing waiting-message.
    It's intended to be used while a task with a separate thread is running. 

    Args:
        wxFrame (wx.Frame): Frame to bind the panel.
        wSz (tuple): Size of the wxFrame.
        msg (str): Text to display in the center of the panel.
        imgFP (str): Image file path.
        bgCol (tuple): Background color of panel.
        wmSz (str): size of panel (full/ small/ tiny)

    Returns:
        None
    """
    if DEBUG: MyLogger.info(str(locals()))

    def onPaint(event, panel):
        event.Skip()
        dc = wx.PaintDC(panel)
        dc.Clear()
        ### draw a frame at the edge of the panel
        pw, ph = panel.GetSize()
        thck = 2
        dc.SetPen(wx.Pen((255,255,0), thck)) 
        dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
        dc.DrawRectangle(thck, thck, pw-thck*2, ph-thck*2)
    # make panel
    if wmSz == "full": pos = (0, 0)
    else: pos = (5, 5)
    panel = wx.Panel(wxFrame, -1, pos=pos, size=wSz)
    panel.SetBackgroundColour(bgCol)
    panel.Bind(wx.EVT_PAINT, lambda event: onPaint(event, panel)) 

    ### show icon image
    if wmSz == "full": aniFP = imgFP.replace(".png", ".gif")
    else: aniFP = imgFP.replace(".png", "_%s.gif"%(wmSz))
    if path.isfile(aniFP):
        icon = Animation(aniFP)
        aCtrl = AnimationCtrl(panel, -1, ani)
        ret = aCtrl.Play()
    else:
        icon = load_img(imgFP) 
        bmp = wx.Bitmap(icon)
        iBmp = wx.StaticBitmap(panel, -1, bmp)

    ### show message in the center of the panel
    _fsz = dict(full=16, small=12, tiny=8)
    font = wx.Font(pointSize=_fsz[wmSz], family=wx.DEFAULT, style=wx.NORMAL,
                   weight=wx.BOLD)
    sTxt = wx.StaticText(panel, -1, label=msg, size=(wSz[0],-1))
    sTxt.SetFont(font)
    fgCol = getConspicuousCol(bgCol)
    sTxt.SetForegroundColour(fgCol)

    ### set positions & panel size
    gap = 20
    dc = wx.MemoryDC()
    dc.SetFont(font)
    tw, th = dc.GetTextExtent(msg)
    iw, ih = icon.GetSize()
    if wmSz == "full": 
        tPos = (int(wSz[0]/2 - tw/2), int(wSz[1]/2 - th/2)) 
        iPos = (tPos[0]-iw-gap, int(wSz[1]/2-ih/2))
    else:
        tPos = (iw+gap, int(ih/2)+gap)
        iPos = (gap, gap)
        pSz = (tw+iw+gap*3, ih+gap*2)
        panel.SetSize(pSz)
        panel.SetPosition((int(wSz[0]/2-pSz[0]/2), pos[1]))
    sTxt.SetPosition(tPos)
    if path.isfile(aniFP): aCtrl.SetPosition(iPos)
    else: iBmp.SetPosition(iPos)
    
    panel.Refresh() 
    wxFrame.Refresh()
    wxFrame.tmpWaitingMsgPanel = panel

#-------------------------------------------------------------------------------

def showStatusBarMsg(wxFrame, txt, delTime=5000, txtBgCol="#33aa33"):
    """ Show message on status bar

    Args:
        wxFrame (wx.Frame)
        txt (str): Text to show on status bar.
        delTime (int): Duration (in milliseconds) to show the text.
        txtBgCol (str): Color to change when displaying text. 

    Returns:
        None
    """
    if DEBUG: MyLogger.info(str(locals()))

    if wxFrame.timer["sb"] != None:
        ### stop status-bar timer
        wxFrame.timer["sb"].Stop()
        wxFrame.timer["sb"] = None
    
    # show text on status bar 
    wxFrame.statusbar.SetStatusText(txt)
    
    ### change status bar color
    if txt == '': bgCol = wxFrame.sbBgCol 
    else: bgCol = txtBgCol 
    wxFrame.statusbar.SetBackgroundColour(bgCol)
    wxFrame.statusbar.Refresh()

    if txt != '' and delTime != -1:
    # showing message and deletion time was given.
        # schedule to delete the shown message
        wxFrame.timer["sb"] = wx.CallLater(delTime, showStatusBarMsg,
                                           wxFrame, '')

#-------------------------------------------------------------------------------

def procExceptionWX(wxFrame, e, msg, closingFlag):
    """ Process an exception in a wxPython app.

    Args:
        wxFrame (wx.Frame)
        e (Exception)
        msg (str): Message to show.
        closingMsg (str): Flag string to hand it over to 'onClose' function. 

    Returns:
        None
    """ 
    if DEBUG: MyLogger.info(str(locals()))

    wx.MessageBox(msg, "ERROR", wx.OK|wx.ICON_ERROR)
    em = ''.join(traceback.format_exception(None, e, e.__traceback__))
    print(em)
    wx.CallLater(10, wxFrame.onClose, closingFlag)

#-------------------------------------------------------------------------------
    
def wxSndPlay(snd="", asyn=True):
    """ Play sound with wx.adv.Sound 

    Args:
        snd (str/wx.adv.Sound): Path to the sound file to play or 
                                already loaded sound
        asyn (bool): Whether asynchronous sound play or not. 

    Returns:
        None
    """ 
    if DEBUG: MyLogger.info(str(locals()))

    if type(snd) == str: snd = wx.adv.Sound(snd)
    if asyn: snd.Play(wx.adv.SOUND_ASYNC)
    else: snd.Play(wx.adv.SOUND_SYNC)

#-------------------------------------------------------------------------------

def getConspicuousCol(col): 
    """ get a color, which is conspicuous (mostly complment color) 
    from the given RGB color

    Args:
        col (tuple): RGB color tuple

    Returns:
        (tuple): Tuple of RGB values 
    """ 
    if DEBUG: MyLogger.info(str(locals()))
    
    col = [col[x]/255.0 for x in range(3)]
    hsv = colorsys.rgb_to_hsv(col[0], col[1], col[2])
    h = (hsv[0] + 0.5) % 1
    s = hsv[1]
    v = hsv[2]
    if (not (0.6 < hsv[0] < 0.7)) or (hsv[1] < 0.2) or (hsv[2] < 0.2):
    # non-blue color OR the color is close to black or white
        v = 1.0 - hsv[2]
    tmp = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    return tmp

#-------------------------------------------------------------------------------

def convt_idx_to_ordinal(number):
    """ Convert zero-based index number to ordinal number string
    0->1st, 1->2nd, ...

    Args:
        number (int): An unsigned integer number.

    Returns:
        (str): Converted string

    Examples:
        >>> convt_idx_to_ordinal(0)
        '1st'
    """
    if DEBUG: MyLogger.info(str(locals()))
    
    if number == 0: return "1st"
    elif number == 1: return "2nd"
    elif number == 2: return "3rd"
    else: return "%ith"%(number+1)

#-------------------------------------------------------------------------------

def convt_180_to_360(ang):
    """ convert -180~180 degree to 0~360 degree

    Args:
        angle (int): Output angle, 0 indicates right, 90 indicates up, 
            180 indicates left, -90 indicates down
        
    Returns:
       angle (int): Input angle. 0 indicates right, 90 indicates up,
            180 indicates left, 270 indicates down. 
    
    Examples:
        >>> convt_180_to_360(45)
        45
        >>> convt_180_to_360(-45)
        315
    """ 
    if DEBUG: MyLogger.info(str(locals()))
    if ang < 0: ang = 360 + ang
    return ang

#-------------------------------------------------------------------------------

def convt_360_to_180(angle):
    """ Convert 360 degree system to 180 degree system.

    Args:
        angle (int): Input angle. 0 indicates right, 90 indicates up,
            180 indicates left, 270 indicates down.

    Returns:
        angle (int): Output angle, 0 indicates right, 90 indicates up, 
            180 indicates left, -90 indicates down

    Examples:
        >>> convt_360_to_180(90)
        90
        >>> convt_360_to_180(270)
        -90
    """
    if DEBUG: MyLogger.info(str(locals()))
    if angle <= 180: return angle
    else: return -(360 % angle)

#-------------------------------------------------------------------------------

def calc_pt_w_angle_n_dist(angle, dist, bPosX=0, bPosY=0, flagScreen=False):
    """ Calculates a point when a angle and a distance is given.

    Args:
        angle (int): 0 indicates right, 90 indicates up, 
            180 or -180 indicates left, -90 indicates down.
        dist (int): Distance in pixel.
        bPosX (int): x-coordinate of base-position.
        bPosY (int): y-coordinate of base-position.
        flagScreen (bool): whether it's for displaying it on screen.
          (y coordinate will be reversed)

    Returns:
        (int, int): x,y coordinate

    Examples:
        >>> calc_pt_w_angle_n_dist(90, 20)
        (0, 20)
        >>> calc_pt_w_angle_n_dist(180, 20)
        (-20, 0)
        >>> calc_pt_w_angle_n_dist(-135, 20)
        (-14, -14)
        >>> calc_pt_w_angle_n_dist(-135, 20, 100, 100)
        (85, 85)
        >>> calc_pt_w_angle_n_dist(-135, 20, 100, 100, True)
        (85, 114)
    """
    if DEBUG: MyLogger.info(str(locals()))

    s = np.sin(np.deg2rad(angle))
    c = np.cos(np.deg2rad(angle))
    x = int(bPosX + c*dist)
    if flagScreen: y = int(bPosY - s*dist)
    else: y = int(bPosY + s*dist)
    return (x, y)

#-------------------------------------------------------------------------------

def calc_line_angle(pt1, pt2):
    """ Calculates angle of a line, defined with two points (pt1 and pt2)

    Args:
        pt1 (tuple): x, y coordinate of the first point
        pt2 (tuple): x, y coordinate of the second point

    Returns:
        (int): angle of the line; 0=right, 90=upward, -90=downward, 180=left

    Examples:
        >>> calc_line_angle((0,0), (1,0))
        0
        >>> calc_line_angle((0,0), (1,1))
        -45
        >>> calc_line_angle((0,0), (-1,-1))
        135
    """
    if DEBUG: MyLogger.info(str(locals()))
    return int(np.degrees(np.arctan2(-(pt2[1]-pt1[1]),pt2[0]-pt1[0])))

#-------------------------------------------------------------------------------

def calc_angle_diff(ang1, ang2):
    """ Calculates angle difference between two angles

    Args:
        ang1 (int): Angle between -180 and 180
        ang2 (int): Angle between -180 and 180

    Returns:
        (int): Angle difference (smallest)

    Examples:
        >>> calc_angle_diff(0, 90)
        90
        >>> calc_angle_diff(180, 45)
        135
        >>> calc_angle_diff(180, -90)
        90
    """
    if DEBUG: MyLogger.info(str(locals()))
    angle_diff = 0
    if (ang1 >= 0 and ang2 >= 0) or (ang1 < 0 and ang2 < 0):
        angle_diff = abs(ang1-ang2)
    elif (ang1 >=0 and ang2 < 0) or (ang1 < 0 and ang2 >= 0):
        ad1 = abs(ang1) + abs(ang2)
        ad2 = 180-abs(ang1) + 180-abs(ang2)
        angle_diff = min(ad1, ad2)
    return angle_diff

#-------------------------------------------------------------------------------

def calc_pt_dists(arr):
    """ calculate distances of all points on all other points 
          in a (Nx2) array of points.

    Args:
        arr (numpy.ndarray): (N x 2) array of (x,y) points. 

    Returns:
        dist (numpy.ndarray): (N x N) array of distances 
    """ 
    if DEBUG: MyLogger.info(str(locals()))
    
    return np.linalg.norm(arr-arr[:,None], axis=-1)

#-------------------------------------------------------------------------------

def calc_pt_line_dist(pt, line, flag_line_ends=True):
    """ Calculates distance from a point to a line

    Args:
        pt (tuple): a point
        line (tuple): a line, defined with two points; ((x1,y1), (x2,y2))
        flag_line_ends : whether line ends at (x1,y1) & (x2,y2) 
            or indefinitely extends

    Returns:
        (float): the distance between point and line

    Examples:
        >>> calc_pt_line_dist((0,0), ((1,0), (0,1)))
        0.7071067811865476
        >>> calc_pt_line_dist((0,0), ((1,0), (0.5,0.25)), True)
        0.5590169943749475
        >>> calc_pt_line_dist((0,0), ((1,0), (0.5,0.25)), False)
        0.4472135954999579
    """
    if DEBUG: MyLogger.info(str(locals()))
    lpt1 = line[0]; lpt2 = line[1]
    ldx = lpt2[0]-lpt1[0]
    ldy = lpt2[1]-lpt1[1]
    sq_llen = ldx**2 + ldy**2 # square length of line
    if sq_llen == 0: # line is a point
        return np.sqrt( (pt[0]-lpt1[0])**2 + (pt[1]-lpt1[1])**2 )
    u = ( (pt[0]-lpt1[0])*ldx + (pt[1]-lpt1[1])*ldy ) / float(sq_llen)
    x = lpt1[0] + u * ldx
    y = lpt1[1] + u * ldy
    if flag_line_ends:
        if u < 0.0: x, y = lpt1[0], lpt1[1] # beyond lpt1-end of segment
        elif u > 1.0: x, y = lpt2[0], lpt2[1] # beyond lpt2-end of segment
    dx = pt[0] - x
    dy = pt[1] - y
    return np.sqrt(dx**2 + dy**2)

#-------------------------------------------------------------------------------

def rot_pt(pt, ct, deg):
    """ Rotate (counter-clockwise) point;pt, around center point;ct
    * y-coordinate follows computer screen coordinate system,
    ssssdhere 0 is the top row and the row index increases as it comes down

    Args:
        pt (tuple): Point to rotate
        ct (tuple): Center point
        deg (float): Angle to rotate

    Returns:
        (tuple): Rotated point. 
    
    Examples:
        >>> rot_pt((2,2), (1,1), 45)
        (2, 1)
        >>> rot_pt((2,2), (1,1), 180)
        (0, 0)
        >>> rot_pt((2,2), (1,1), -90)
        (0, 2)
    """ 
    if DEBUG: MyLogger.info(str(locals()))

    r = np.deg2rad(deg)
    tx = pt[0]-ct[0]
    ty = pt[1]-ct[1]
    x = (tx * np.cos(r) + ty * np.sin(r)) + ct[0]
    y = (-tx * np.sin(r) + ty * np.cos(r)) + ct[1]
    return (int(np.round(x)), int(np.round(y)))

#-------------------------------------------------------------------------------

def convertSagittaArc2AngArc(pt1, pt2, sagitta):
    """ Convert arc (circular arc) info (two points and bulge) to 
    another type of arc info (center-point, radius, angle for pt1 and 
    angle for pt2)
    - code of Alexander Reynolds; http://reynoldsalexander.com/

    Args:
        pt1 (tuple): Point-1 
        pt2 (tuple): Point-2
        sagitta (int): Positive or negative integer to indicate bulge distance

    Returns:
        (tuple): Center point
        radius (int): Radius
        pt1Ang (int): pt1 angle
        pt2Ang (int): pt2 angle
        (tuple): the 3rd point (middle of arc)

    Examples:
        >>> ret = convertSagittaArc2AngArc(pt1, pt2, sagitta)
        >>> axes = (ret[1], ret[1])
        >>> modCV.drawEllipse(img, ret[0], axes, 0, ret[2], ret[3], color=255)
    """ 
    if DEBUG: MyLogger.info(str(locals()))

    x1, y1 = pt1
    x2, y2 = pt2

    ### calculate the 3rd point (the point on arc, where sagitta distance
    ###   away from the middle point of the line between pt1 and pt2)
    n = np.array([y2 - y1, x1 - x2])
    n_dist = np.sqrt(np.sum(n**2))
    if np.isclose(n_dist, 0):
        print('Error: The distance between pt1 and pt2 is too small.')
    n = n/n_dist
    x3, y3 = (np.array(pt1) + np.array(pt2))/2 + sagitta*n

    ### calculate the circle from three points
    ###   see https://math.stackexchange.com/a/1460096/246399
    ###   or http://web.archive.org/web/20161011113446/http://www.abecedarical.com/zenosamples/zs_circle3pts.html
    A = np.array([
        [x1**2 + y1**2, x1, y1, 1],
        [x2**2 + y2**2, x2, y2, 1],
        [x3**2 + y3**2, x3, y3, 1]])
    M11 = np.linalg.det(A[:, (1, 2, 3)])
    M12 = np.linalg.det(A[:, (0, 2, 3)])
    M13 = np.linalg.det(A[:, (0, 1, 3)])
    M14 = np.linalg.det(A[:, (0, 1, 2)])
    if np.isclose(M11, 0):
        print('Error: The third point is collinear. (sagitta ~ 0)')
    cx = 0.5 * M12/M11
    cy = -0.5 * M13/M11
    radius = np.sqrt(cx**2 + cy**2 + M14/M11)

    ### calculate angles of pt1 and pt2 from center of circle
    pt1Ang = 180*np.arctan2(y1 - cy, x1 - cx)/np.pi
    pt2Ang = 180*np.arctan2(y2 - cy, x2 - cx)/np.pi

    return (cx, cy), radius, pt1Ang, pt2Ang, (x3, y3)

#-------------------------------------------------------------------------------

def receiveDataFromQueue(q, logFile=''):
    """ Receive data from a queue.

    Args:
        q (Queue): Queue to receive data.
        logFile (str): File path of log file.

    Returns:
        rData (): Data received from the given queue. 

    Examples:
        >>> receiveDataFromQueue(Queue(), 'log.txt')
    """
    if DEBUG: MyLogger.info(str(locals()))

    rData = None
    try:
        if q.empty() == False: rData = q.get(False)
    except Exception as e:
        em = "%s, [ERROR], %s\n"%(get_time_stamp(), str(e))
        if path.isfile(logFile) == True: writeFile(logFile, em)
        print(em)
    return rData    

#-------------------------------------------------------------------------------

def csv2numpyArr(csvTxt, delimiter, numericDataIdx, stringDataIdx,
                 npNumericDataType):
    """ convert CSV text from the csv file text to Numpy array of 
    numeric & string data.
    
    Args:
        csvTxt (str): Read text from CSV file.
        delimiter (str): Seperator for CSV data.
        numericDataIdx (list): List of column indices of numerica data.
        stringDataIdx (list): List of column indices of string data.
        npNumericDataType (numpy.dtype): Data type for numpy array.
    
    Returns:
        colTitles (list): Column titles
        numData (numpy.ndarray): Numeric data array.
        strData (nummpy.char.array): String data array.
       
    Examples:
        >>> ret = csv2numpyArr(csvTxt, ",", [1,2,3], [0], np.uint8)
    """ 
    if DEBUG: MyLogger.info(str(locals()))

    colTitles = []  # data column titles
    numData = [] # numeric data
    strData = [] # string data
    lines = csvTxt.split("\n")
    for line in lines:
        items = [x.strip() for x in line.split(delimiter)]

        while "" in items: items.remove("") # remove empty data
        if len(items) <= 1: continue # ignore emtpy line.
          # line with one item is just a comment line. Also ignore.
        
        ### store column titles
        # The first proper (len(items)>1) line is assumed to be
        # the title line
        if colTitles == []:
            for ii in range(len(items)):
                colTitles.append(items[ii])
            continue 

        ### store data
        rsd = [] # row of string data
        rnd = [] # row of numeric data
        for ci in range(len(colTitles)): # through columns
            val = items[ci].strip()
            if ci in numericDataIdx:
                val = str2num(val)
                if val == None: val = -1 
                rnd.append(val) # add numeric data
            elif ci in stringDataIdx:
                val = val.replace(" ","") # remove blanks
                rsd.append(val) # add string data
        ### add a data line
        if rnd != []: numData.append(rnd)
        if rsd != []: strData.append(rsd)
    
    if numData != []: numData = np.asarray(numData, npNumericDataType)
    if strData != []: strData = np.char.asarray(strData)
    
    return colTitles, numData, strData

#-------------------------------------------------------------------------------

def getNumpyDataType(num, flagIntSign=False):
    """ return a proper numpy data type for the given number

    Args:
        num (float/integer): a number value
        flagIntSign (bool): Unsigned integer or signed integer

    Returns:
        (NumPy data type)
    """
    if DEBUG: MyLogger.info(str(locals()))

    if "." in str(num): # float values
        if num < np.finfo(np.float16).max: return np.float16
        else: return np.float32
    else: # integer values
        if flagIntSign:
            if num < np.iinfo(np.int8).max: return np.int8
            elif num < np.iinfo(np.int16).max: return np.int16
            else: return np.int32
        else:
            if num < np.iinfo(np.uint8).max: return np.uint8
            elif num < np.iinfo(np.uint16).max: return np.uint16
            else: return np.uint32

#-------------------------------------------------------------------------------

def getPercentileValsNPos(arr):
    """ Get values of percentiles (0, 25, 50, 75, 100) and 
    its positions in the given array 

    Args:
        arr (list): Input array

    Returns:
        pv (list): Values at percentiles; 0, 25, 50, 75, 100 
        pvPos (list): Indices of the each 'pv' value in the 'arr'
    """
    if DEBUG: MyLogger.info(str(locals()))

    pv = [] 
    pvPos = [] 
    for p in range(0, 101, 25):
        _pv = np.percentile(arr, p)
        pv.append(_pv)
        # find the index of the nearest value
        pvPos.append((np.abs(arr-_pv)).argmin())
    
    return pv, pvPos

#-------------------------------------------------------------------------------

#===============================================================================

class TableBase(wx.grid.GridTableBase):
    def __init__(self, data):
        if DEBUG: MyLogger.info(str(locals()))
        wx.grid.GridTableBase.__init__(self)
        self.data = data
        self.colLabels = list(range(len(data.dtype)))
        # frame indices
        self.rowLabels = [str(x) for x in range(data.shape[0])]
    
    #---------------------------------------------------------------------------

    def GetNumberRows(self): 
        #if DEBUG: MyLogger.info(str(locals()))
        return self.data.shape[0] 

    #---------------------------------------------------------------------------

    def GetNumberCols(self): 
        #if DEBUG: MyLogger.info(str(locals()))
        return len(self.data.dtype)

    #---------------------------------------------------------------------------

    def GetValue(self, row, col):
        #if DEBUG: MyLogger.info(str(locals()))
        return self.data[row][col]

    #---------------------------------------------------------------------------

    def SetValue(self, row, col, value):
        #if DEBUG: MyLogger.info(str(locals()))
        self.data[row][col] = value
    
    #---------------------------------------------------------------------------
    
    def GetRowLabelValue(self, row):
        #if DEBUG: MyLogger.info(str(locals()))
        return self.rowLabels[row]
    
    #---------------------------------------------------------------------------
    
    def SetRowLabelValue(self, row, value):
        #if DEBUG: MyLogger.info(str(locals()))
        self.rowLabels[row] = value
    
    #---------------------------------------------------------------------------
    
    def GetColLabelValue(self, col):
        #if DEBUG: MyLogger.info(str(locals()))
        return self.colLabels[col]
    
    #---------------------------------------------------------------------------

    def SetColLabelValue(self, col, value):
        #if DEBUG: MyLogger.info(str(locals()))
        self.colLabels[col] = value

    #---------------------------------------------------------------------------

#===============================================================================

class Grid(wx.grid.Grid): 
    def __init__(self, parent, data, size=(100,50)): 
        if DEBUG: MyLogger.info(str(locals()))
        wx.grid.Grid.__init__(self, parent, -1, size=size) 
        self.table = TableBase(data) 
        self.SetTable(self.table, True) 

#===============================================================================

class STC(wx.stc.StyledTextCtrl):
    def __init__(self, parent, pos, size, fgCol="#000000", bgCol="#999999",
                 caretFGCol="#ffffff"):
        if DEBUG: MyLogger.info(str(locals()))
        
        lineNumFGCol = bgCol 
        lineNumBGCol = fgCol
        wx.stc.StyledTextCtrl.__init__(self, parent, -1, pos=pos, size=size,
                                       style=wx.SIMPLE_BORDER)
        self.StyleClearAll()
        self.SetViewWhiteSpace(wx.stc.STC_WS_VISIBLEALWAYS)
        self.SetIndentationGuides(1)
        self.SetViewEOL(0)
        self.SetIndent(4) 
        self.SetMarginType(0, wx.stc.STC_MARGIN_NUMBER)
        self.SetMarginWidth(0, self.TextWidth(0,'9999'))
        self.StyleSetSpec(wx.stc.STC_STYLE_DEFAULT, 
                          "face:Monaco,fore:%s,back:%s,size:12"%(fgCol, bgCol))
        self.StyleSetSpec(wx.stc.STC_P_DEFAULT, 
                          "face:Monaco,fore:%s,back:%s,size:12"%(fgCol, bgCol))
        self.StyleSetSpec(
          wx.stc.STC_STYLE_LINENUMBER, 
          "face:Monaco,fore:%s,back:%s,size:12"%(lineNumFGCol, lineNumBGCol)
          )
        self.SetCaretStyle(2)
        self.SetCaretForeground(caretFGCol)
        self.StyleSetBackground(wx.stc.STC_STYLE_DEFAULT, bgCol) 
        self.SetSTCCursor(0) # To make cursor as a pointing arrow. 
                             # Can't find reference for STC_CURSOR, 
                             # but '1' resulted in the text-input cursor.

#===============================================================================

class OutlierDetection_GESD:
    """ for detecting outlier in 1D data (should be frmo normal distribution),
    using 'Generalized Extreme Studentized Deviate'.
    Adjusted code from the code written by Shaleen Swarup.
    
    Args:
        arr (numpy.ndarray): Input array to calculate.
        alpha (float): Significance level.
        maxOutliers (float): Indicates maximum number of outliers. 
            Ratio of number of outliers to the length of input date, 'arr'.
        verbose (bool): Whether to print additional info.
    """
    def __init__(self, arr, alpha=0.05, maxOutliers=0.1, verbose=False):
        if DEBUG: MyLogger.info(str(locals()))
        
        ##### [begin] setting up attributes -----
        self.arr = arr # input array
        self.alpha = alpha
        self.nMaxOutliers = round(len(self.arr)* maxOutliers)
        self.verbose = verbose
        ##### [end] setting up attributes ----- 

    #---------------------------------------------------------------------------

    def testStat(self, y, iteration):
        """ calculates test statistic

        Args:
            y (np.ndarray): input array.
            iteration (int): iteration number in GESD.

        Returns:
            cal (float): calculated statistic.
            maxIdx (int): index of data with max statistic.
        """
        if DEBUG: MyLogger.info(str(locals()))

        std_dev = np.std(y)
        avg_y = np.mean(y)
        abs_val_minus_avg = abs(y - avg_y)
        max_of_deviations = max(abs_val_minus_avg)
        maxIdx = np.argmax(abs_val_minus_avg)
        cal = max_of_deviations/ std_dev
        if self.verbose:
            print('Test {}'.format(iteration))
            print("Test Statistics Value(R{}) : {}".format(iteration,cal))
        return cal, maxIdx 
    
    #---------------------------------------------------------------------------

    def calcCriticalValue(self, nInput, iteration):
        """ calculates critical value of t-distribution

        Args:
            nInput (int): Length of input data.
            iteration (int): iteration number in GESD.

        Returns:
            critical_value (float): calculated critical value. 
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        t_dist = stats.t.ppf(1-self.alpha / (2*nInput), nInput-2)
        numerator = (nInput-1) * np.sqrt(np.square(t_dist))
        denominator = np.sqrt(nInput) * np.sqrt(nInput-2+np.square(t_dist))
        critical_value = numerator / denominator
        if self.verbose:
            print("Critical Value(λ{}): {}".format(iteration, critical_value))
        return critical_value
    
    #---------------------------------------------------------------------------

    def detect(self):
        """ Detect outlier using 

        Args:
            None

        Returns:
            olIdx (list): List of outlier indices (in original input array).
            isNormal (bool): Whether it's from normal distribution after 
                removing all outliers.
        """
        if DEBUG: MyLogger.info(str(locals()))

        inputArr = copy(self.arr)
        statLst = []
        criticalLst = []
        olIdx = [] # outlier indices in original array
        isNormal = True
        if self.verbose:
            print("[begin] outlier detection -----")
        for iteration in range(1, self.nMaxOutliers+1):
            stat, maxIdx = self.testStat(inputArr, iteration)
            critical = self.calcCriticalValue(len(inputArr), iteration)
            
            if stat > critical: # outlier
                _idx = int(np.where(self.arr==inputArr[maxIdx])[0])
                olIdx.append(_idx)
            else: # not outlier any more
                break

            if self.verbose:
                msg = "{} is ".format(inputArr[maxIdx])
                if stat <= critical: msg += "not "
                msg += "an outlier. "
                msg += "R{} > λ{}: ".format(iteration, iteration)
                msg += "{:.4f} > {:.4f}\n".format(stat, critical)
                print(msg)

            inputArr = np.delete(inputArr, maxIdx)
            criticalLst.append(critical)
            statLst.append(stat)
            if stat > critical:
                max_i = iteration

        stat, p = stats.normaltest(inputArr)
        msg = "\nThe array after removing outliers is "
        thr = 1e-2
        if p < 1e-2:
            msg += "not "
            isNormal = False
        msg += "from normal distribution; p={}, alpha={}\n".format(p, thr)

        if self.verbose:
            print(msg)
            msg = "\nH0: there are no outliers in the data\n"
            msg += "Ha: there are up to 10 outliers in the data\n"
            msg += "Significance level: α = {}\n".format(self.alpha)
            msg += "Critical region:  Reject H0 if Ri > critical value\n"
            msg += "Ri: Test statistic\n"
            msg += "λi: Critical Value\n\n"
            msg += "Number of outliers {}\n".format(max_i)
            msg += "[end] outlier detection -----"
            print(msg)
    
        return isNormal, olIdx
    
    #---------------------------------------------------------------------------

#===============================================================================

class PopupDialog(wx.Dialog):
    """ Class for showing a message to a user.
    This class was made to use it as a base class for a dialog box
      with more widgets such as a dialog box to enter some info.
    
    Args:
        parent (wx.Frame): Parent object (probably, wx.Frame or wx.Panel).
        id (int): ID of this dialog.
        title (str): Title of the dialog.
        msg (str): Message to show.
        iconFP (str): File path of an icon image.
        font (wx.Font): Font of message string.
        pos (None/ tuple): Position to make the dialog window.
        size (tuple): Size of dialog window.
        flagOkayBtn (bool): Whether to show Ok button.
        flagNoBtn (bool): Whether to show No button.
        flagCancelBtn (bool): Whether to show Cancel button.
        flagDefOK (bool): Whether Ok button has focus by default (so that 
          user can just press enter to dismiss the dialog window).
        bgColor (str): Background color.
        addW (list): Widget info when there're additional widgets to show,
                  instead of a simple message.
    """
    def __init__(self, 
                 parent=None, 
                 id=-1, 
                 title="Message", 
                 msg="", 
                 iconFP="", 
                 font=None, 
                 pos=None, 
                 size=None,
                 flagOkayBtn=True, 
                 flagNoBtn=False, 
                 flagCancelBtn=False, 
                 flagDefOK=False,
                 bgColor="#333333",
                 addW=[]):
        if DEBUG: MyLogger.info(str(locals()))

        ### init Dialog
        wx.Dialog.__init__(self, parent, id, title) 
        if size is None: size = (int(parent.wSz[0]*0.3), int(parent.wSz[1]*0.3))
        self.SetSize(size)
        if pos is None: self.Center()
        else: self.SetPosition(pos)
        # init panel
        self.panel = dict(mp = SPanel.ScrolledPanel(self, 
                                                    -1, 
                                                    pos=(0,0),
                                                    size=size))
        if sys.platform.startswith("win"):
            c = copy(bgColor)
            if type(bgColor) == str:
                c = c.lstrip("#")
                c = [int(c[x:x+2], 16) for x in (0, 2, 4)]
            bgColor = tuple([min(x+50, 255) for x in c])
        self.panel["mp"].SetBackgroundColour(bgColor)

        ### font setup 
        if font == None:
            font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.NORMAL, 
                           wx.FONTWEIGHT_NORMAL, False, "Arial", 
                           wx.FONTENCODING_SYSTEM)

        ##### [begin] set up widgets -----
        self.gbs = {}
        bw = 15 
        w = [] # widge list; each item represents a row in the panel 
        nCol = 3 
        if msg.strip() != "":
            msgW = int(size[0]*0.7)
            ### display message
            if iconFP != "" and path.isfile(iconFP):
                bmp = wx.Bitmap(load_img(iconFP))
                bmpSz = bmp.GetSize()
                msgW -= bmpSz[0]
                w.append([
                            {"type":"sBmp", "name":"icon", "nCol":1, 
                             "bitmap":bmp, "size":bmpSz, "border":bw},
                            {"type":"sTxt", "label":msg, "nCol":1, "font":font,
                             "wrapWidth":msgW, "border":bw},
                            {"type":"sTxt", "label":" ", "nCol":1}
                            ])
            else:
                w.append([
                            {"type":"sTxt", "label":msg, "nCol":3, "font":font,
                             "wrapWidth":msgW, "fgColor":"#ffffff",
                             "border":bw}
                            ])

        # add additional widget info
        w += addW

        ### okay button
        lineItem = []
        if flagOkayBtn:
            lineItem.append({"type":"btn", "name":"ok", "nCol":1, "id":wx.ID_OK,
                             "label":"Okay", "size":(100,-1), "border":bw})
        else:
            lineItem.append({"type":"sTxt", "label":" ", "nCol":1})
        ### No button
        if flagNoBtn:
            lineItem.append({"type":"btn", "name":"no", "nCol":1, 
                             "id":wx.ID_NO, "label":"No",
                             "size":(100,-1), "border":bw})
        else:
            lineItem.append({"type":"sTxt", "label":" ", "nCol":1})
        ### cancel button
        if flagCancelBtn:
            lineItem.append({"type":"btn", "name":"cancel", "nCol":1, 
                             "id":wx.ID_CANCEL, "label":"Cancel",
                             "size":(100,-1), "border":bw})
        else:
            lineItem.append({"type":"sTxt", "label":" ", "nCol":1})
        w.append(lineItem)
        ### 
        self.gbs["mp"] = wx.GridBagSizer(0,0) 
        widLst, pSz = addWxWidgets(w, self, "mp") 
        self.panel["mp"].SetSizer(self.gbs["mp"])
        self.gbs["mp"].Layout()
        self.panel["mp"].SetupScrolling()
        ##### [end] set up widgets -----
        if flagOkayBtn:
            if flagCancelBtn == False or flagDefOK == True:
                self.panel["mp"].Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
                btn = wx.FindWindowByName("ok_btn", self)
                btn.SetDefault()
      
    #---------------------------------------------------------------------------

    def onKeyPress(self, event):
        """ Process key-press event
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        if event.GetKeyCode() == wx.WXK_RETURN: 
            self.EndModal(wx.ID_OK)
    
    #---------------------------------------------------------------------------
    
    def getValues(self, wNames):
        """ Return values of widgets 
        
        Args:
            wNames (list): List of widget names to retrieve its value 

        Returns:
            values (dict): retrieved values 
        """
        if DEBUG: MyLogger.info(str(locals()))

        values = {} 
        
        for wn in wNames:
            key, wTyp = wn.split("_")
            w = wx.FindWindowByName(wn, self.panel["mp"])
            if wTyp in ["txt", "spin", "chk", "sld"]:
                values[key] = w.GetValue()
            elif wTyp in ["cho", "radB"]:
                values[key] = w.GetString(w.GetSelection())
            elif wTyp == "cPk":
                values[key] = tuple(w.GetColour()[:3])
        
        return values
    
#===============================================================================

if __name__ == '__main__':
    pass
