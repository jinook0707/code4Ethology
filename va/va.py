# coding: UTF-8

"""
This is the main Python file for an open-source program 
 for detecting certain visual features such as a presence of an animal.

Coded and tested on Ubuntu 22.04 

Jinook Oh, Acoustics Research Institute, Wien, Austria.
last edited: 2024-10-27

Dependency:
    Python (3.x)
    wxPython (4.0)
    OpenCV (4.7)
    SciPy (1.1)
    filterpy (1.4)

------------------------------------------------------------------------
Copyright (C) 2024 Jinook Oh & Marisa Hoeschele 
- Contact: jinook0707@gmail.com/ jinook.oh@oeaw.ac.at

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------

Changelog
------------------------------------------------------------------------
v.0.1; 2024.08
    - Init.
"""

import sys, queue, traceback
from threading import Thread 
from os import getcwd, path, mkdir
from copy import copy, deepcopy
from time import time, sleep
from datetime import timedelta
from glob import glob
from random import shuffle

import wx
import wx.lib.scrolledpanel as SPanel 
import cv2
import numpy as np

sys.path.append("..")
import initVars
initVars.init(__file__)
from initVars import *

from modCV import VideoRW
from modFFC import *
from procFrames import ProcFrames

FLAGS = dict(
                debug = False,
                seqFR2jump = False,
                log = True,
                )
MyLogger = setMyLogger("VideoAnalysis")
__version__ = "0.1.202408"

#===============================================================================

class VAFrame(wx.Frame):
    """ Frame for Video Analysis 
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self):
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ### init frame
        if sys.platform.startswith("win"): wPos = (0,0)
        else: wPos = [0, 25]
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], int(wg[3]*0.85))
        title = "VideoAnalysis v.%s"%(__version__)
        style = wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX)
        wx.Frame.__init__(self, None, -1, title, 
                          pos=tuple(wPos), size=tuple(wSz), style=style)
        self.SetBackgroundColour("#333333")
        ### set app icon
        iconPath = path.join(P_DIR, "image", "icon.tif")
        if __name__ == '__main__' and path.isfile(iconPath):
            self.SetIcon(wx.Icon(iconPath)) # set app icon
        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        # frame resizing
        updateFrameSize(self, (wSz[0], wSz[1]+self.statusbar.GetSize()[1]))
        # frame close event
        self.Bind(wx.EVT_CLOSE, self.onClose)
        
        ##### [begin] setting up attributes -----
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread 
        self.wPos = wPos # window position
        self.wSz = wSz # window size
        self.fonts = getWXFonts(initFontSz=8, numFonts=3)
        pi = self.setPanelInfo()
        self.pi = pi # pnael information
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for statusbar
        self.session_start_time = -1
        self.runningDur = 0 # accumulated duration (sec.) of continuous running
        self.logFP = path.join(FPATH, "log.txt")
        writeFile(self.logFP, "\n", mode="w") # delete log from prev. session
        self.oData = [] # output data; index of this list = frame-index
        self.inputFPLst = [] # input file list (when folder was opened)
        self.inputFP = "" # input file path
        self.outputFP = "" # output folder path
        self.flags = {}
        self.flags["blockUI"] = False # block user input 
        self.flags["isRunning"] = False # is video analysis currently running
        self.allowedVideoExt = ["mp4", "mkv", "mov", "avi"] 
        # animal experiment cases
        self.animalECaseChoices = [
                "i202408", # default case 
                ]
        ### set display image type choices
        self.dispImgTypeChoices = {}
        self.dispImgTypeChoices["i202408"] = [
            "frame", "grayscale", "prevgray", "blur", "fg", 
            "color", "color2", "color3", "motion" 
                                ]
        self.animalECase = "i202408"
        # current display image type
        self.dispImgType = self.dispImgTypeChoices[self.animalECase][0]
        i202408Str = "Default case"
        self.aecHelpStr = dict(
                                    i202408=i202408Str,
                               ) 
        self.lpWid = [] # wx widgets in left panel
        self.ratFImgDispImg = None # ratio between frame image and 
          # display image on app
        self.analyzedFrame = None # for storing frame image after CV analysis
        self.procF = ProcFrames(self) # frame-processing with CV
        self.vRW = VideoRW(self) # video reading/writing 
        # set ouput data columns (self.dataCols),
        #   initival values (self.dataInitVal) and column indices 
        self.dataCols, self.dataInitVal, self.di = self.procF.setDataCols()  
        self.setParams() # set parameters for analysis
        ##### [end] setting up attributes -----

        vlSz = (-1, 20) # vertical line size
        btnSz = (35, 35)
        btnBGCol = "#333333"
        ### create panels
        for pk in pi.keys():
            w = [] # widge list; each item represents a row in the panel 
            if pk == "tp": # top panel
                w.append([ 
                    {"type":"sTxt", "label":"Experiment case: ", "nCol":1,
                     "border":10, "fgColor":"#cccccc"},
                    {"type":"cho", "nCol":1, "name":"animalECase", 
                     "choices":self.animalECaseChoices, "val":self.animalECase,
                     "border":5},
                    {"type":"btn", "name":"aec:help", "nCol":1,
                     "img":path.join(P_DIR,"image","help.png"),
                     "tooltip":"Description of the chosen experiment case",
                     "size":btnSz, "bgColor":btnBGCol, "border":0},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"btn", "name":"openDir", "nCol":1,
                     "img":path.join(P_DIR,"image","open.png"),
                     "tooltip":"Open a folder with videos", "size":btnSz,
                     "bgColor":btnBGCol, "border":5},
                    {"type":"btn", "name":"open", "nCol":1,
                     "img":path.join(P_DIR,"image","camera.png"),
                     "tooltip":"Open a video file", "size":btnSz,
                     "bgColor":btnBGCol, "border":5},
                    #{"type":"btn", "name":"analyzeImgs", "nCol":1,
                    # "img":path.join(P_DIR,"image","picture.png"),
                    # "tooltip":"Analyze images", "size":btnSz,
                    # "bgColor":btnBGCol},
                    {"type":"txt", "nCol":1, "val":"", "name":"fp", 
                     "style":wx.TE_READONLY, "size":(200,-1), "border":5},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"btn", "name":"contAnalysis", "nCol":1,
                     "img":path.join(P_DIR,"image","startStop.png"),
                     "tooltip":"Start/Stop analysis", "size":btnSz, 
                     "bgColor":btnBGCol,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":5},
                    {"type":"btn", "name":"nextFrame", "nCol":1,
                     "img":path.join(P_DIR,"image","start1.png"),
                     "tooltip":"Process the next frame",
                     "size":btnSz, "bgColor":btnBGCol,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":5},
                    {"type":"txt", "nCol":1, "val":"1", "name":"jump2frame", 
                     "size":(75,-1), "numOnly":True, "border":0},
                    {"type":"btn", "name":"jump2frame", "nCol":1,
                     "img":path.join(P_DIR,"image","jump.png"),
                     "tooltip":"Move to the selected frame",
                     "size":btnSz, "bgColor":btnBGCol, "border":0}, 
                    {"type":"btn", "nCol":1, "name":"save", "size":btnSz,
                     "img":path.join(P_DIR, "image", "save.png"), 
                     "tooltip":"Save data", "bgColor":btnBGCol},
                    {"type":"sTxt", "label":"", "nCol":1, "border":10,
                     "flag":wx.RIGHT},
                    {"type":"btn", "nCol":1, "name":"delData", "size":btnSz,
                     "img":path.join(P_DIR, "image", "deleteAll.png"), 
                     "tooltip":"Delete all data calculated so far", 
                     "bgColor":btnBGCol},
                    {"type":"sTxt", "label":"", "nCol":1, "border":10,
                     "flag":wx.RIGHT},
                    {"type":"chk", "nCol":1, "name":"budgieVideoClip",
                     "label":"clip-videos-with-budgies", "style":wx.CHK_2STATE,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":20,
                     "val":False, "fgColor":"#cccccc"},
                    {"type":"sTxt", "label":"", "nCol":1, "border":20,
                     "flag":wx.RIGHT},
                    {"type":"sTxt", "label":"0:00:00", "nCol":1,
                     "name":"ssTime", "fgColor":"#ccccff", 
                     "bgColor":"#000000"},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"sTxt", "label":"FPS", "nCol":1, "name":"fps",
                     "fgColor":"#cccccc"}, 
                    ])
            setupPanel(w, self, pk, True)

        lastWidget = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
        x = lastWidget.GetPosition()[0] + lastWidget.GetSize()[0]
        if x > wSz[0]:
        # last widget of the top panel is over the window width
            ### resize panels (for scroll bar in top panel)
            if sys.platform.startswith("win"): inc = 30
            else: inc = 15
            tpSz = self.pi["tp"]["sz"]
            tpSz = (tpSz[0], tpSz[1]+inc)
            self.panel["tp"].SetSize(tpSz)
            for pk in self.pi.keys():
                if pk == "tp": continue
                pos = self.pi[pk]["pos"]
                self.panel[pk].SetPosition((pos[0], pos[1]+inc))
                sz = self.pi[pk]["sz"]
                self.panel[pk].SetSize((sz[0], sz[1]-inc))

        self.initLPWidgets() # left panel widgets

        ### bind events to image panel 
        self.panel["rpI"].Bind(wx.EVT_PAINT, self.onPaintIP)
        self.panel["rpI"].Bind(wx.EVT_LEFT_DOWN, self.onMLBDown)
        self.panel["rpI"].Bind(wx.EVT_LEFT_UP, self.onMLBUp)
        self.panel["rpI"].Bind(wx.EVT_MOTION, self.onMMove)
       
        # set up menu
        self.setUpMenuBar()

        self.log("Program starting")

    #---------------------------------------------------------------------------

    def setUpMenuBar(self):
        """ set up the menu bar

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        menuBar = wx.MenuBar()
        ### VA menu
        menu = wx.Menu()
        openV = menu.Append(wx.Window.NewControlId(), item="Open video\tCTRL+O")
        self.Bind(wx.EVT_MENU, lambda event: self.openInputData(False), openV)

        openF = menu.Append(wx.Window.NewControlId(), 
                            item="Open folder\tCTRL+F")
        self.Bind(wx.EVT_MENU, lambda event: self.openInputData(True), openF)
        saveR = menu.Append(wx.Window.NewControlId(), item="Save\tCTRL+S")
        self.Bind(wx.EVT_MENU, self.save, saveR)

        quitA = menu.Append(wx.Window.NewControlId(), item="Quit\tCTRL+Q")
        self.Bind(wx.EVT_MENU, self.onClose, quitA)

        menuBar.Append(menu, "&VA")

        ### Help menu
        menuH = wx.Menu()
        helpS = menuH.Append(wx.Window.NewControlId(), item="Help string\tF1")
        self.Bind(wx.EVT_MENU,
                  lambda event: self.onButtonPressDown(event, "aec:help_btn"),
                  helpS)
        menuBar.Append(menuH, "&Help")
        
        self.SetMenuBar(menuBar) 

    #---------------------------------------------------------------------------
   
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
         
        Returns:
            pi (dict): Panel information.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        wSz = self.wSz # window size
        #style = (wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        #style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
        style = (wx.DEFAULT_FRAME_STYLE)
        pi = {} # panel information to return
        if sys.platform.startswith("win"): hght = 50
        else: hght = 55
        # top panel for buttons, etc
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], hght), bgCol="#666666", 
                        style=style)
        tpSz = pi["tp"]["sz"]
        if sys.platform.startswith("win"): width = int(wSz[0]*0.21)
        else: width = int(wSz[0]*0.12)
        # left side panel for setting parameters
        pi["lp"] = dict(pos=(0, tpSz[1]), bgCol="#333333", style=style,
                        sz=(width, wSz[1]-tpSz[1]))
        lpSz = pi["lp"]["sz"]
        
        # panel for displaying frame image 
        pi["rpI"] = dict(pos=(lpSz[0], tpSz[1]), sz=(wSz[0]-lpSz[0], lpSz[1]), 
                         bgCol="#000000", style=style)
        
        return pi
    
    #---------------------------------------------------------------------------
    
    def initLPWidgets(self):
        """ initialize wx widgets in left panel

        Args: None

        Returns: None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        for i, w in enumerate(self.lpWid):
        # through widgets in left panel
            self.gbs["lp"].Detach(w) # detach widget from gridBagSizer
            w.Destroy() # destroy the widget

        ### make a list to produce widgets with parameter keys
        lst = list(self.aecParam.keys()) # list of keys for parameter items 
        uLst = [] # this parameter items will be more likely modified by users.
        for i in range(len(lst)):
            if lst[i][0] == 'u':
                uLst.append(str(lst[i]))
                lst[i] = None
        lst = list(filter(None, lst))
        lst = sorted(uLst) + ["---"] + sorted(lst) 

        ##### [begin] set up left panel interface -----
        nCol = 5
        wSz = self.wSz
        lpSz = self.pi["lp"]["sz"]
        btnBGCol = "#333333"
        w = [] # list of widgets
        dispImgTypeChoices = self.dispImgTypeChoices[self.animalECase]
        self.dispImgType = dispImgTypeChoices[0]
        w.append([
            {"type":"sTxt", "label":"Display-image", "nCol":1,
             "font":self.fonts[1], "fgColor":"#cccccc"},
            {"type":"cho", "nCol":nCol-1, "name":"imgType",
             "choices":dispImgTypeChoices, "size":(160,-1),
             "val":self.dispImgType}
            ]) 
        w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":1}])
        for key in lst:
            if key == "---":
                w.append([{"type":"sTxt", "label":"", "nCol":nCol, 
                           "border":10}])
                continue
            param = self.aecParam[key]
            if "disableUI" in param.keys(): disable = param["disableUI"]
            else: disable = False
            w.append([
                # static-text to show name of the parameter
                {"type":"sTxt", "label":key, "nCol":1, "font":self.fonts[0],
                 "fgColor":"#cccccc", "border":1},
                # help button to describe what the parameter is about
                {"type":"btn", "label":"?", "nCol":1, "name":"%s:help"%(key),
                 "size":(25,-1), "fgColor":"#ffffff", "bgColor":btnBGCol, 
                 "border":1},
                {"type":"sTxt", "label":"   ", "nCol":1, "font":self.fonts[0],
                 "border":1},
                # parameter value
                {"type":"sld", "nCol":1, "name":key, "val":param["value"], 
                 "size":(int(wSz[0]*0.1),-1), "style":wx.SL_VALUE_LABEL, 
                 "minValue":param["minVal"], "maxValue":param["maxVal"], 
                 "fgColor":"#ffffff", "border":1, "disable":disable}
                ])
            
            if (key.startswith("uROI") and key.lower().endswith("y")):
                    w.append([{"type":"sTxt", "nCol":nCol, "label":" "}])

        w.append([{"type":"sTxt", "nCol":nCol, "label":" "}])
        _ttS = "Save parameters as a file, for the currently opened video"
        w.append([
            {"type":"btn", "nCol":nCol, "name":"saveParam", "tooltip":_ttS, 
             "bgColor":btnBGCol, "fgColor":"#ffffff", "label":"Save param.", 
             "border":10}
            ])
        w.append([{"type":"sTxt", "nCol":nCol, "label":" "}])
        
        self.lpWid = setupPanel(w, self, "lp")
        ##### [end] set up left panel interface -----

        w = wx.FindWindowByName(f'{lst[-1]}_sld', self.panel["lp"])
        lpRX = w.GetPosition()[0] + int(w.GetSize()[0] * 1.25)
        if lpRX > lpSz[0]:
        # the slider widget is over the panel width
            ### resize panels 
            xInc = lpRX - lpSz[0]
            for pk in self.panel.keys():
                if pk == "tp": continue
                if pk == "lp":
                    self.pi[pk]["sz"] = (lpRX, lpSz[1])
                elif pk == "rpI":
                    _pos = self.pi[pk]["pos"]
                    self.pi[pk]["pos"] = (_pos[0]+xInc, _pos[1])
                    _sz = self.pi[pk]["sz"]
                    self.pi[pk]["sz"] = (_sz[0]-xInc, _sz[1])
                if pk != "lp": self.panel[pk].SetPosition(self.pi[pk]["pos"])
                self.panel[pk].SetSize(self.pi[pk]["sz"])

        ### re-size the save parameter button 
        w = wx.FindWindowByName('saveParam_btn', self.panel["lp"])
        wx.CallLater(300, w.SetSize, int(lpRX*0.9), -1)

    #---------------------------------------------------------------------------

    def config(self, flag):
        """ saving/loading configuration of the current animal-experiment-case 

        Args:
            flag (str): save or load

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        configFP = path.join(FPATH, f'config_{self.animalECase}')

        if flag == "save":
            fh = open(configFP, "wb")
            pickle.dump(self.aecParam, fh)
            fh.close()

        elif flag == "load":
            if path.isfile(configFP):
            # config file exists
                fh = open(configFP, "rb")
                config = pickle.load(fh)
                fh.close()
                for k in config.keys():
                    if k not in self.aecParam.keys():
                        if k.startswith("uROI"):
                            self.aecParam[k] = dict(
                                value=config[k]["value"], minVal=0, maxVal=100, 
                                desc="Region-of-interest (%%)"
                                )
                        continue
                    self.aecParam[k]["value"] = config[k]["value"]
            else:
                return False

        return True 

    #---------------------------------------------------------------------------
    
    def setParams(self):
        """ Set up parameters for analysis.
        * Parameter key starts with a letter 'u' means that 
          it will probably modified by users more frequently.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.animalECase == "": return
        
        aec = self.animalECase
        self.aecParam = {} # parameter dictionary

        ##### [begin] common parameters -----
        d = "[cv2.MORPH_OPEN]\nNumber of iterations of morphologyEx for"
        d += "  reducing noise & minor visual features in removeDetails"
        d += " function. Higher number means more removal."
        d += " The function is used to obtain a greyscale"
        d += " image for background-extraction and motion-detection."
        self.aecParam["rdMExOIter"] = dict(value=2, minVal=-1, maxVal=10, 
                                           desc=d)
        d = d.replace("[cv2.MORPH_OPEN]", "[cv2.MORPH_CLOSE]")
        self.aecParam["rdMExCIter"] = dict(value=-1, minVal=-1, maxVal=10, 
                                           desc=d) 

        d = "Parameter for cv2.threshold to make the image clear after"
        d += " reducing noise & minor visual features in removeDetails"
        d += " function. The function is used to obtain a greyscale"
        d += " image for background-extraction and motion-detection."
        self.aecParam["rdThres"] = dict(value=-1, minVal=-1, maxVal=254, desc=d)

        d = "Parameters for hysteresis threshold of Canny function"
        d += " (in edge detection of difference from background image)."
        d += " Currently (2024-05-13), the edge detection is not used."
        self.aecParam["cannyThMin"] = dict(value=50, minVal=0, maxVal=255, 
                                           desc=d)
        self.aecParam["cannyThMax"] = dict(value=200, minVal=0, maxVal=255,
                                           desc=d)

        d = "Threshold in cv2.threshold function for motion-detection."
        d += " Lower number means more sensitive detection, but more noise."
        self.aecParam["motionThr"] = dict(value=50, minVal=0, maxVal=255, 
                                          desc=d)

        d = "Lower and upper contour area limit (pixels)"
        d += " for one contour of detected motion.\n"
        self.aecParam["motionCntThrMin"] = dict(value=10, minVal=1, 
                                                maxVal=100, desc=d) 
        self.aecParam["motionCntThrMax"] = dict(value=1000, minVal=1, 
                                                maxVal=518400, desc=d) 
        ##### [end] common parameters -----

        # add additional parameter for each experiment case
        self.aecParam = self.procF.initAECaseParam(self.aecParam)

        # load and overload the aecParam values (if config exists)
        ret = self.config("load")

    #---------------------------------------------------------------------------
    
    def onPaintIP(self, event):
        """ painting image panel 

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.inputFP == "": return

        event.Skip()
        dc = wx.PaintDC(self.panel["rpI"])
        dc.SetBackground(wx.Brush((0,0,0,255)))
        dc.Clear()

        if self.analyzedFrame is not None:
            ### display analyzed frame image 
            img = cv2.cvtColor(self.analyzedFrame, cv2.COLOR_BGR2RGB)
            wxImg = wx.Image(img.shape[1], img.shape[0])
            wxImg.SetData(img.tobytes())
            dc.DrawBitmap(wxImg.ConvertToBitmap(), 0, 0)
    
    #---------------------------------------------------------------------------

    def onButtonPressDown(self, event, objName=""):
        """ wx.Butotn was pressed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate the button press
              of the button with the given name. 
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"]: return 

        if objName == '':
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
            obj = wx.FindWindowByName(objName, self.panel["tp"])

        if not obj.IsEnabled(): return

        if objName == "open_btn":
            self.openInputData()

        elif objName == "openDir_btn":
            self.openInputData(isOpeningDir=True)

        elif objName == "save_btn":
            self.save(None)

        elif objName == "delData_btn":
            self.delData()
        
        elif objName == "contAnalysis_btn":
            self.onSpace(None)
        
        elif objName == "quit_btn":
            self.onClose(None)
        
        elif objName == "nextFrame_btn":
            self.onRight(True)
        
        elif objName == "jump2frame_btn":
            txt = wx.FindWindowByName("jump2frame_txt", self.panel["tp"])
            self.jumpToFrame(int(txt.GetValue()))

        elif objName.startswith("applyParam"):
            wn = objName.lstrip("applyParam").replace("_btn","") + "_sld"
            w = wx.FindWindowByName(wn, self.panel["lp"])
            self.applyChangedParam(wn, w.GetValue())
        
        elif objName == "saveParam_btn":
            self.saveParam(None)
        
        elif objName.endswith(":help_btn"):
            key = objName.replace(":help_btn", "")
            msg = ""
            if key == "aec":
                if self.animalECase in self.aecHelpStr.keys():
                    msg = self.aecHelpStr[self.animalECase]
            else:
                msg = self.aecParam[key]["desc"]
            self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)  

    #---------------------------------------------------------------------------

    def onChoice(self, event, objName=""):
        """ wx.Choice was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.Choice event 
                with the given name. 
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "animalECase_cho":
        # animal experiment case was chosen
            self.animalECase = objVal
            self.setParams() # set parameters for analysis
            ret = self.procF.setDataCols() 
            self.dataCols, self.dataInitVal, self.di = ret
            self.initLPWidgets() # initialize left panel
        
        if objName == "imgType_cho":
        # display image type changed
            self.dispImgType = objVal
            self.procImg()

    #---------------------------------------------------------------------------
    
    def onSlider(self, event, objName=""):
        """ wx.Slider was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        self.applyChangedParam(objName, objVal)

    #---------------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

    #---------------------------------------------------------------------------
    
    def onTextCtrlChar(self, event, objName="", isNumOnly=False):
        """ Character entered in wx.TextCtrl.
        Currently using to allow entering numbers only.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name.
            isNumOnly (bool): allow number entering only.
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if isNumOnly:
            keyCode = event.GetKeyCode()
            ### Allow numbers, backsapce, delete, left, right
            ###   and tab (for hopping between TextCtrls)
            allowed = [ord(str(x)) for x in range(10)]
            allowed += [wx.WXK_BACK, wx.WXK_DELETE, wx.WXK_TAB]
            allowed += [wx.WXK_LEFT, wx.WXK_RIGHT]
            if keyCode in allowed:
                event.Skip()
                return

    #---------------------------------------------------------------------------

    def onKeyPress(self, event, kc=None, mState=None):
        """ Process key-press event

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        event.Skip()
        if kc == None: kc = event.GetKeyCode()
        if mState == None: mState = wx.GetMouseState()

        if kc == wx.WXK_SPACE:
            self.onSpace(None)

        #elif kc == wx.WXK_RIGHT:
        #    self.onRight(True)

    #---------------------------------------------------------------------------
       
    def onRight(self, isLoadingOneFrame=False):
        """ Navigate forward 
        
        Args:
            isLoadingOneFrame (bool): whether load only one frame, 
              False by defualt.

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or self.inputFP == "": return
        
        if isLoadingOneFrame: self.flags["isRunning"] = False 

        if self.flags["isRunning"]: # if continuous analysis is running 
            ### FPS update
            self.fps += 1
            if time()-self.last_fps_time >= 1:
                sTxt = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
                sTxt.SetLabel( "FPS: %i"%(self.fps) )
                self.fps = 0
                self.last_fps_time = time()
                self.runningDur += 1
      
        if self.flags["isRunning"] and self.vRW.fi >= self.vRW.nFrames-1:
        # continuous running reached the end of frames
            self.onSpace(None) # stop continuous running

        if self.vRW.fi >= self.vRW.nFrames-1: return
        self.procF.updatePrevGrayImg()
        ret = self.vRW.getFrame() # read one frame

        if not ret and self.flags["isRunning"]: 
            
            if self.vRW.fi >= self.vRW.nFrames*0.9:
            # Read most of frames. Assume that it reached the end of video.
            # (for when nFrames from cv2.CAP_PROP_FRAME_COUNT was inaccurate)
                self.flags["isRunning"] = False
                # cut off the rest of output data 
                self.oData = self.oData[:self.vRW.fi] 
                # process saving, etc..
                self.procAtEndOfVideo()

        else:
            self.procImg() # process the frame
        
    #---------------------------------------------------------------------------

    def onSpace(self, event):
        """ start/stop continuous frame analysis

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or self.inputFP == "": return

        if self.flags["isRunning"] == False:
            if self.vRW.fi >= self.vRW.nFrames-1: return
            self.flags["isRunning"] = True
            self.fps = 0
            self.last_fps_time = time()
            self.timer["run"] = wx.CallLater(1, self.onRight)
        else:
            try: # stop timer
                self.timer["run"].Stop() 
                self.timer["run"] = None
            except:
                pass
            sTxt = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
            sTxt.SetLabel('')
            self.flags["isRunning"] = False # stop continuous analysis
            
    #---------------------------------------------------------------------------
    
    def jumpToFrame(self, targetFI=-1):
        """ leap forward to a frame
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or targetFI == -1 or self.inputFP == "":
            return
        
        if targetFI >= self.vRW.nFrames: targetFI = self.vRW.nFrames-1
        
        if FLAGS["seqFR2jump"]: # sequential frame reading to jump
            # difference between current frame-index and target-frame-index 
            self.diff_FI_TFI= abs(self.vRW.fi - targetFI)
            ret = self.vRW.getFrame(targetFI, useCAPPROP=False, 
                                    callbackFunc=self.callback)
        else:
            ret = self.vRW.getFrame(targetFI, useCAPPROP=True)

        if ret:
            # delete previous grey image
            self.procF.updatePrevGrayImg(flagNone=True)
            self.procImg() # process current frame
            if FLAGS["seqFR2jump"]: # sequential frame reading to jump
                if path.isfile(self.inputFP): self.flags["blockUI"] = True

    #---------------------------------------------------------------------------
    
    def onMLBDown(self, event):
        """ Mouse left button is pressed on displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        if pk != "rpI": return
        
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

        if not self.flags["isRunning"]: # if analysis is not actively running
            ### store the current mouse pointer position 
            if self.ratFImgDispImg != None:
                r = 1.0/self.ratFImgDispImg
                self.panel[pk].mousePressedPt = (int(mp[0]*r), int(mp[1]*r))
            else:
                self.panel[pk].mousePressedPt = (mp[0], mp[1])
        
    #---------------------------------------------------------------------------

    def onMMove(self, event):
        """ Mouse pointer is moving in displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        #if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        if pk != "rpI": return
        
        pos0 = self.panel[pk].mousePressedPt
        if pos0 == (-1, -1): return
        
        mp = event.GetPosition()
        
        ### set pos1, if the displayed frame is a resized image,
        if self.ratFImgDispImg != None:
            r = 1.0/self.ratFImgDispImg
            pos1 = (int(mp[0]*r), int(mp[1]*r))
        else:
            pos1 = (mp[0], mp[1])
     
        '''
        ### NOT being used yet
        mInput = None
        if self.animalECase == "[TBD]": 
            mInput = dict([TBD]v0=pos0[0], [TBD]v0=pos0[1],
                          [TBD]v1=pos1[0], [TBD]v1=pos1[1])
        if mInput is not None: self.procImg(mInput)
        '''
    
    #---------------------------------------------------------------------------

    def onMLBUp(self, event):
        """ Mouse left button was clicked on displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        if pk != "rpI": return

        pos0 = self.panel[pk].mousePressedPt
        if pos0 == (-1, -1): return
        
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()
        self.panel[pk].mousePressedPt = (-1, -1)
        self.procImg()

    #---------------------------------------------------------------------------
    
    def applyChangedParam(self, wn, wVal):
        """ Apply changed parameters.

        Args:
            wn (str): Name of the slider widget.
            wVal (int): Value of the slider.
        
        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.inputFP == "": return

        key = wn.replace("_sld","")
        
        # no data has changed
        if wVal == self.aecParam[key]["value"]: return 

        # update the parameter value
        self.aecParam[key]["value"] = wVal

        # delete previous grey image
        self.procF.updatePrevGrayImg(flagNone=True)

        self.procImg() # process image
        self.config("save") # save config

    #---------------------------------------------------------------------------
    
    def loadParam(self):
        """ Read parameters from a file, if it exists, and store it.

        Args: None
        
        Returns: None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        if self.inputFP == "": return False
        
        ext = "." + self.inputFP.split(".")[-1]
        fp = self.inputFP.replace(ext, "_params.txt")
        if not path.isfile(fp): return False

        fh = open(fp, "r")
        lines = fh.readlines()
        fh.close()

        for line in lines:
            items = line.split(",")
            if len(items) < 2: continue
            key = items[0].strip()
            val = items[1].strip()
            if not key in self.aecParam.keys(): continue
            self.aecParam[key]["value"] = str2num(val)

        return True

    #---------------------------------------------------------------------------
    
    def saveParam(self, event):
        """ Save parameters as a file for the currently opened video. 

        Args: event (wx.Event)
        
        Returns: None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.inputFP == "":
            msg = "There's no opened video file."
            self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
            return

        ext = "." + self.inputFP.split(".")[-1]
        fp = self.inputFP.replace(ext, "_params.txt")
        fh = open(fp, "w")
        for k in self.aecParam.keys():
            val = str(self.aecParam[k]["value"])
            if val[0] == "[": # this is a list
                val = val.replace(",", "/")
            line = "%s, %s\n"%(str(k), val)
            fh.write(line)
        fh.close()
        msg = "Parameters are stored in '%s'."%(fp)
        self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)
    
    #---------------------------------------------------------------------------
    
    def procImg(self, mInput=None):
        """ Process image with cv_proc module, update resultant data.
        
        Args:
            mInput (None/dict): Manual user input such as mouse click & drag.

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        if self.inputFP == '' or len(self.oData) == 0: return

        vFI = self.vRW.fi # video frame index
        
        ### set temporary (for processing the current frame)
        ###   dictionary to store values
        tD = {} # temp. dictionary
        for dIdx, dCol in enumerate(self.dataCols):
            if mInput != None and dCol in mInput.keys():
            # manual input is given
                tD[dCol] = mInput[dCol]
                flagMP = True
            elif self.oData[vFI][dIdx] != 'None':
            # already calculated data available
                tD[dCol] = self.oData[vFI][dIdx]
            else:
                tD[dCol] = self.dataInitVal[dIdx]
            ### convert value to integer if applicable
            try: tD[dCol] = int(tD[dCol])
            except: pass
            ### data from previous frame
            pk = "p_" + dCol
            if vFI == 0:
                tD[pk] = self.dataInitVal[dIdx] 
            else:
                tD[pk] = self.oData[vFI-1][dIdx]
                if not tD[pk] in ['None', 'D', 'True', 'False']:
                    try: tD[pk] = int(tD[pk])
                    except: pass
       
        try:
            with wx.BusyInfo("Processing...", None):
                # process frame image
                frame, retD = self.procF.procImg(self.vRW.currFrame.copy(), 
                                                  tD)
                wx.Yield()
        except Exception as e:
            _str = "".join(traceback.format_exc())
            print(_str)
            self.showMsg(_str, "ERROR", wx.OK|wx.ICON_ERROR)
            return

        # display the processed frame 
        if self.ratFImgDispImg != 1.0:
            w = int(frame.shape[1] * self.ratFImgDispImg)
            h = int(frame.shape[0] * self.ratFImgDispImg)
            frame = cv2.resize(frame, (w, h))
        self.analyzedFrame = frame
        self.panel["rpI"].Refresh()

        ### update oData
        if retD != None:
            for dIdx, dCol in enumerate(self.dataCols):
                self.oData[vFI][dIdx] = str(retD[dCol])

        if self.flags["isRunning"]:
            if vFI < self.vRW.nFrames-1: # there's more frames to run
                # continue to analyse
                self.timer["run"] = wx.CallLater(5, self.onRight)
            else: # reached end of the frames
                self.procAtEndOfVideo() 

    #---------------------------------------------------------------------------
     
    def procAtEndOfVideo(self):
        """ Process when reached end of video data. 

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        self.save(None) # save result data

        self.procF.postProcVideoRunning(self.oData)

        if self.inputFPLst != []:
        # not a single video, but a folder was opened
            self.inputFPIdx += 1
            if self.inputFPIdx >= len(self.inputFPLst):
            # reached end of file list
                self.onSpace(None) # stop continuous running
                msg = "All files are analyzed."
                self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)
            else:
                self.setIOPaths(self.inputFPLst[self.inputFPIdx])
                self.startAnalysis()
        else:
            self.onSpace(None) # stop continuous running

    #---------------------------------------------------------------------------
     
    def initDataWithLoadedVideo(self):
        """ Init. info of loaded video and init/load result data.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        self.loadParam() # load AEC parameters if a param file 
                         #   for the specific video file exists.
        ret = self.procF.setDataCols() 
        self.dataCols, self.dataInitVal, self.di = ret
        self.initLPWidgets() # initialize left panel
        self.oData, endDataIdx = self.initOutputData() # init output data

        self.runningDur = 0
        self.session_start_time = time()
        ### set timer for updating the current session running time
        self.timer["sessionTime"] = wx.Timer(self)
        self.Bind(wx.EVT_TIMER,
                  lambda event: self.onTimer(event, "sessionTime"),
                  self.timer["sessionTime"])
        self.timer["sessionTime"].Start(1000)

        rsltCSVFile = self.inputFP + '.csv'
        if path.isfile(rsltCSVFile) and endDataIdx > 0:
        # result CSV file exists & 
        # there's, at least, one data (with head direction) exists 
            self.jumpToFrame(endDataIdx) # move to the 1st None value

        txt = wx.FindWindowByName("fp_txt", self.panel["tp"])
        txt.SetValue('%s'%(path.basename(self.inputFP)))

        self.log(f'Starting to analyze a file; {self.inputFP}')

        self.procImg() # process current frame

        showStatusBarMsg(self, "", -1)

    #---------------------------------------------------------------------------
    
    def initOutputData(self):
        """ initialize output data

        Args: None

        Returns: None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        rsltCSVFile = self.inputFP + '.csv'
        oData = []
        if path.isfile(rsltCSVFile):
        # if there's previous result file for this video
            for fi in range(self.vRW.nFrames):
                oData.append(list(self.dataInitVal))
            # load previous CSV data
            oData, endDataIdx = self.loadData(rsltCSVFile, oData)
        else:
            for fi in range(self.vRW.nFrames):
                oData.append(list(self.dataInitVal))
            endDataIdx = fi
        # return output data as NumPy character array
        #return np.asarray(oData, dtype=self.dataStruct), endDataIdx
        return oData, endDataIdx

    #---------------------------------------------------------------------------
     
    def loadData(self, rsltCSVFile, oData):
        """ Load data from CSV file

        Args:
            rsltCSVFile (str): Result CSV file name.
            oData (list): Output data.

        Returns:
            oData (list): Output data with loaded data.
            endDataIdx (int): Index of last data.
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        endDataIdx = -1 
        ### read CSV file and update oData
        f = open(rsltCSVFile, 'r')
        lines = f.readlines()
        f.close()
        for li in range(1, len(lines)):
            items = [x.strip() for x in lines[li].split(',')]
            if len(items) <= 1: continue
        
            ### restore parameters
            if items[0] == 'AEC':
                self.animalECase = items[1]
                continue
            if items[0] in self.aecParam.keys():
                val = items[1].strip("[]").split("/")
                for vi in range(len(val)):
                    try: val[vi] = int(val[vi])
                    except:
                        try: val[vi] = float(val[vi])
                        except: pass
                if len(val) == 1: val = val[0]
                self.aecParam[items[0]]['value'] = val 
                continue
            
            ### restore data
            try: fi = int(items[0])
            except: continue
            for ci in range(len(self.dataCols)):
                val = str(items[ci+1])
                oData[fi][ci] = val
                if endDataIdx == -1:
                    if ci == 0 and val == 'None':
                    # store frame-index of 'None' value in the 1st column 
                        endDataIdx = copy(fi) 
            # make row as a tuple for structured numpy array
            #oData[fi] = tuple(oData[fi]) 
            oData[fi] = list(oData[fi]) 

        if endDataIdx == -1: endDataIdx = fi 
        return (oData, endDataIdx)

    #---------------------------------------------------------------------------
    
    def openInputData(self, isOpeningDir=False):
        """ Open input data file/folder. 

        Args:
            isOpeningDir (bool): Opening folder with input videos.

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        animalECase_cho = wx.FindWindowByName("animalECase_cho", 
                                              self.panel["tp"])
        if self.session_start_time == -1:
        # not in analysis session. start a session
            if self.inputFP == '':
                ### open input file (folder)
                defDir = path.join(FPATH, "data")
                if isOpeningDir:
                    t = "Choose directory for analysis"
                    style = (wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST) 
                    dlg = wx.DirDialog(self, t, defaultPath=defDir, style=style)
                else:
                    t = "Open video file"
                    _wc = ""
                    for ext in self.allowedVideoExt:
                        _wc += "*.%s;"%(ext.lower())
                        _wc += "*.%s;"%(ext.upper())
                    _wc = _wc.rstrip(";")
                    wc = " (%s)|%s"%(_wc, _wc)
                    style = (wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
                    dlg = wx.FileDialog(self, t, defaultDir=defDir, 
                                        wildcard=wc, style=style)
                if dlg.ShowModal() == wx.ID_CANCEL:
                    dlg.Destroy()
                    return
                ##### [begin] get input video file path(s) -----
                if isOpeningDir:
                    dirPath = dlg.GetPath()
                    self.inputFPLst = []
                    ### add file paths of mp4 or avi files
                    fsLst = []
                    for ext in self.allowedVideoExt:
                        fsLst.append("*.%s"%(ext))
                        fsLst.append("*.%s"%(ext.upper()))
                    for fs in fsLst:
                        _path = path.join(dirPath, fs)
                        for fp in glob(_path):
                            if not fp in self.inputFPLst:
                                self.inputFPLst.append(fp)
                    ### remove video file paths without parameter text file
                    for i, fp in enumerate(self.inputFPLst):
                        ext = "." + fp.split(".")[-1]
                        paramFP = fp.replace(ext, "_params.txt")
                        if not path.isfile(paramFP):
                        # parameter text file doesn't exist
                            self.inputFPLst[i] = None # remove
                    self.inputFPLst = list(filter(None, self.inputFPLst))
                    self.inputFPLst.sort(key=natural_keys)
                    if len(self.inputFPLst) == 0:
                    # no video file to analyze 
                        msg = "No video file with parameter file is found."
                        self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
                        return
                    ### get user confirmation on list of files 
                    msg = "* Selected folder:\n  %s\n\n"%(dirPath)
                    msg += "* List of video (mp4 or avi) files, "
                    msg += " found with parameter text files:\n"
                    for fp in self.inputFPLst:
                        msg += "  %s\n"%(path.basename(fp))
                    msg = msg.rstrip(",")
                    msg += "\n\nPlease confirm the list of files.\n"
                    msg += "After pressing the okay button here and the start"
                    msg += " button on the top panel, the program will go"
                    msg += " through all files. Each time one file is finished,"
                    msg += " a result CSV file will be generated in the same"
                    msg += " folder with the same name to the video file"
                    sz = (int(self.wSz[0]/2), int(self.wSz[1]/2))
                    dlg = PopupDialog(self, title="Query", msg=msg, 
                                      font=self.fonts[1], size=sz,
                                      flagCancelBtn=True) 
                    if dlg.ShowModal() == wx.ID_CANCEL:
                        dlg.Destroy()
                        return
                    ### set the first input file
                    self.inputFPIdx = 0
                    self.setIOPaths(self.inputFPLst[self.inputFPIdx])
                else:
                    self.setIOPaths(dlg.GetPath())
                ##### [end] get input video file path(s) -----
                dlg.Destroy()
            showStatusBarMsg(self, "Processing... please wait.", -1)
            animalECase_cho.Disable()  
            self.startAnalysis() 
            
        else: # in session. stop it.
            '''
            dlg = PopupDialog(self, 
                              title="Query", 
                              msg="Save data?", 
                              flagCancelBtn=True)
            rslt = dlg.ShowModal()
            dlg.Destroy()
            if rslt == wx.ID_OK:
                self.save(None) # save data
            '''
            if self.flags["isRunning"]:
                self.onSpace(None) # stop continuous running
            self.procF.bg = None # remove background image
            self.vRW.closeReader() # close video
            ### init
            sTxt = wx.FindWindowByName("ssTime_sTxt", self.panel["tp"])
            sTxt.SetLabel("0:00:00")
            txt = wx.FindWindowByName("fp_txt", self.panel["tp"])
            txt.SetLabel("")
            self.session_start_time = -1
            self.inputFPLst = [] 
            self.inputFP = ""
            self.outputFP = ""
            self.oData = []
            self.analyzedFrame = None
            animalECase_cho.Enable()
            self.panel["rpI"].Refresh()
            wx.CallLater(10, self.openInputData, isOpeningDir)

    #---------------------------------------------------------------------------
    
    def setIOPaths(self, inputFP):
        """ set output folder path 

        Args:
            inputFP (str): Input file path

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        self.inputFP = inputFP 
        self.outputFP = inputFP + "_output"
        if not path.isdir(self.outputFP): mkdir(self.outputFP)

    #---------------------------------------------------------------------------
    
    def startAnalysis(self):
        """ Start analysis with an input video file. 

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        self.vRW.closeReader()        
        self.vRW.initReader(self.inputFP) # load video file to analyze

        ### calculate ratio to resize to display it in UI
        iSz = (self.vRW.currFrame.shape[1], self.vRW.currFrame.shape[0])
        self.ratFImgDispImg = calcI2DRatio(iSz, self.pi["rpI"]["sz"])
        
        self.procF.initOnLoading() # init variables on loading data
        
        ext = "." + self.inputFP.split(".")[-1]
        fp = self.inputFP.replace(ext, "_params.txt")
        if path.isfile(fp):
        # parameter file for this input video already exists
            self.initDataWithLoadedVideo()
        else:
        # parameter file does not exist
            ### start pre-processing 
            args = (self.q2m,)
            startTaskThread(self, "preProc", 
                            self.procF.preProcess, args=args)
    
    #---------------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread

        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        postProcTaskThread(self, flag)

        """ DEPRECATED;; used for jumping to certain frame index 
                         without using cv2.CAP_PROP_POS_FRAMES
        if flag == "readFrames":
            if self.diff_FI_TFI> 1: # if moving multiple frames
                # update last_motion_frame
                #   to prevent difference goes over motion detection threshold 
                self.procF.last_motion_frame = self.vRW.currFrame.copy()
            self.procImg() # process loaded image
            self.flags["blockUI"] = False
        """

        if flag == "preProc":
            rD = rData[1]
            for k in rD.keys():
                aecp = self.aecParam
                if k == "bgExtract":
                    pass
            
            self.vRW.initReader(self.inputFP) # init video again
            self.initDataWithLoadedVideo()

    #---------------------------------------------------------------------------
    
    def save(self, event):
        """ Saving anaylsis result to CSV file.

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        fp = path.join(self.outputFP, "rslt.csv")
        fh = open(fp, 'w')

        '''
        ### write parameters
        fh.write("Timestamp, %s\n"%(get_time_stamp()))
        fh.write("AEC, %s\n"%(self.animalECase))
        for key in sorted(self.aecParam.keys()):
            val = str(self.aecParam[key]['value'])
            if "," in val: val = val.replace(",", "/")
            line = "%s, %s\n"%(key, val)
            fh.write(line)
        fh.write('-----\n')
        '''

        ### write column heads 
        line = "frame-index, "
        for col in self.dataCols: line += "%s, "%(col)
        line = line.rstrip(", ") + "\n"
        fh.write(line)
        
        ### write data 
        for fi in range(len(self.oData)):
            line = "%i, "%(fi)
            for ci in range(len(self.dataCols)):
                line += "%s, "%(str(self.oData[fi][ci]))
            line = line.rstrip(", ") + "\n"
            fh.write(line)
        fh.write('-----\n')
        fh.close()

        fn = path.basename(fp)
        msg = "Saved file; '%s'\n"%(fn)
        showStatusBarMsg(self, msg, 5000)
        
        msg0 = "Saved a file '%s'"%(fn)
        if self.inputFPLst != []:
            msg0 += "; Finished"
            msg0 += " the %s file"%(convt_idx_to_ordinal(self.inputFPIdx))
            msg0 += " out of %s files"%(len(self.inputFPLst))
        self.log(f'{msg0}')

    #---------------------------------------------------------------------------
    
    def delData(self):
        """ Delete data (init. entire data dict), calculated so far 

        Args: None 

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.inputFP == "": return

        msg = "This will initialize the output data container,"
        msg += " deleting all calculated data so far.\n"
        msg += " Proceed?"
        _sz = (int(self.wSz[0]/3), int(self.wSz[1]/5))
        dlg = PopupDialog(self, title="Query", msg=msg, size=_sz,
                          flagCancelBtn=True) 
        ret = dlg.ShowModal()
        dlg.Destroy()
        if ret == wx.ID_CANCEL: return
        
        for vFI in  range(self.vRW.nFrames):
            for dIdx, dCol in enumerate(self.dataCols):
                self.oData[vFI][dIdx] = self.dataInitVal[dIdx] 
        self.procF.initOnLoading() # init variables
        self.jumpToFrame(targetFI=0)
        
        ### pre-processing 
        args = (self.q2m,)
        startTaskThread(self, "preProc", self.procF.preProcess, args=args)

    #---------------------------------------------------------------------------

    def onTimer(self, event, flag):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        #if FLAGS["debug"]: MyLogger.info(str(locals()))

        if flag == "sessionTime": 
            ### update session running time
            if self.session_start_time != -1 and self.flags["isRunning"]:
                lbl = str(timedelta(seconds=self.runningDur)).split('.')[0]
                sTxt = wx.FindWindowByName("ssTime_sTxt", self.panel["tp"])
                sTxt.SetLabel(lbl)

        else:
            ### Receive data from queue
            ### * if it's "displayMsg", keep receiving until getting 
            ###   the last queued message.
            ### * if the received data is other type than a simple message, 
            ###   then process it.
            rData = None
            while True:
                ret = receiveDataFromQueue(self.q2m)
                if ret == None:
                    break
                else:
                    rData = ret # store received data
                    if ret[0] != "displayMsg": break 
            if rData == None: return
            
            if rData[0] == "displayMsg":
                showStatusBarMsg(self, rData[1], -1)
            
            elif rData[0].startswith("finished"):
                self.callback(rData, flag)
 
    #---------------------------------------------------------------------------
    
    def showMsg(self, msg, mType, flag):
        """ display simple message 

        Args:
            msg (str): Message to show.
            mType (str): Message type.
            flag (int): flags for wx.MessageBox
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        wx.MessageBox(msg, mType, flag)
        self.panel["tp"].SetFocus()

    #---------------------------------------------------------------------------
    
    def log(self, msg, tag="VAFrame"):
        """ leave a log 

        Args:
            msg (str): Message to show.
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if not FLAGS["log"]: return
        msg = f'{get_time_stamp()}, [{tag}], {msg}\n'
        writeFile(self.logFP, msg)

    #---------------------------------------------------------------------------

    def onClose(self, event):
        """ Close this frame.
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        self.log('Program closing')
        
        stopAllTimers(self.timer)
        rslt = True
        if self.flags["isRunning"]: # continuous analysis is running
            msg = "Session is running.\n"
            msg += "Unsaved data will be lost. (Stop analysis or"
            msg += " Cmd+S to save.)\nOkay to proceed to exit?"
            dlg = PopupDialog(self, title="Query", msg=msg, flagCancelBtn=True) 
            if dlg.ShowModal() == wx.ID_CANCEL:
                dlg.Destroy()
                return
            dlg.Destroy()
        
        if hasattr(self.vRW, "video_rec") and self.vRW.video_rec != None:
            self.vRW.closeWriter()
        if sys.platform.startswith("win"): wx.CallLater(200, sys.exit)
        else: wx.CallLater(100, self.Destroy)
    
    #---------------------------------------------------------------------------

#===============================================================================

class VAApp(wx.App):
    def OnInit(self):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        self.frame = VAFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-w", "-c"]:
            if sys.argv[1] == '-w': GNU_notice(1)
            elif sys.argv[1] == '-c': GNU_notice(2)
            sys.exit()

        elif sys.argv[1] == "-jumpBySeqRead":
            FLAGS["seqFR2jump"] = True 

    GNU_notice(0)
    CWD = getcwd()
    app = VAApp(redirect=False)
    app.MainLoop()


