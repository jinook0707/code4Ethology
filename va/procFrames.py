# coding: UTF-8

"""
Computer vision processing on videos for VideoAnalysis program (va.py).

Coded and tested on Ubuntu 22.04.

Jinook Oh, Acoustics Research Institute, Wien, Austria.
last edited: 2024-10-23

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
"""

import itertools, re, subprocess
from time import time, sleep
from datetime import datetime, timedelta
from copy import copy
from glob import glob
from random import randint, choice
from os import path, mkdir

import cv2, pandas
import numpy as np
from scipy.cluster.vq import vq, kmeans 
from scipy.cluster.hierarchy import fclusterdata
# pairwise distances between observations in n-dimensional space. 
from scipy.spatial.distance import pdist 
import torch
import torchvision
from torchvision.transforms import functional as TTFunctional
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as FRCNNv2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as FRCNNv2Wght
from torchvision.models.detection import retinanet_resnet50_fpn_v2 as RNv2 
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights as RNv2Wght
from skimage.feature import graycomatrix, graycoprops 

from initVars import *
from modFFC import * 
from modCV import * 
from procBudgie import ProcBudgieVideo 

FLAGS = dict(
                debug = False,
                )

#===============================================================================

class ProcFrames:
    """ Class for processing a frame image using computer vision algorithms
    to code animal position/direction/behaviour
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ##### [begin] setting up attributes -----
        self.prnt = parent
        self.bg = None # background image of chosen video
        self.fSize = (960, 540) # default frame size
        self.font= cv2.FONT_HERSHEY_SIMPLEX
        self.fScle = 0.5 # default font scale
        self.fThck = 2 # default font thickness
        (self.tW, self.tH), self.tBL = cv2.getTextSize("0", self.font, 
                                                       self.fScle, self.fThck) 

        self.cluster_cols = [
                                (200,200,200), 
                                (255,0,0), 
                                (0,255,0), 
                                (0,0,255), 
                                (255,100,100), 
                                (100,255,100), 
                                (100,100,255), 
                                (0,255,255),
                                (255,0,255), 
                                (255,255,0), 
                                (100,255,255),
                                (255,100,255), 
                                (255,255,100), 
                            ] # BGR color for each cluster in clustering
        self.procMod = None
        self.grayImg = None
        self.prevGrayImg = None
        #self.storage = {} # storage for previsouly calculated parameters 
        #  or temporary frame image sotrage, etc...
        ##### [end] setting up attributes -----

    #---------------------------------------------------------------------------
    
    def initOnLoading(self):
        """ initialize variables when input video was loaded
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt

        self.procMod = None
        self.grayImg = None
        self.prevGrayImg = None
        
        if prnt.animalECase == "i202408":
            self.procMod = ProcBudgieVideo(self, prnt)
             
    #---------------------------------------------------------------------------
    
    def postProcVideoRunning(self, oData):
        """ check/process things after running all the frames
        
        Args:
            oData (list): Output data for each frame
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt
       
        if prnt.animalECase == "i202408":
            ### if video clipping check-box is off, return
            w = wx.FindWindowByName("budgieVideoClip_chk", prnt.panel["tp"])
            chkVideoClip = widgetValue(w)
            if not chkVideoClip: return
            self.procMod.extractVideoClips(oData)

    #---------------------------------------------------------------------------
    
    def preProcess(self, q2m):
        """ pre-process video before running actual analysis 
        to obtain certain data/info. 
        
        Args:
            q2m (None/queue.Queue): Queue to send data to main thread.
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt
        nFrames = prnt.vRW.nFrames
        aecp = prnt.aecParam 
        fH, fW = prnt.vRW.currFrame.shape[:2] # frame height & width
 
        retData = {} # data to return

        if prnt.animalECase in ["i202408"]:
        # certain cases which uses background extraction
            retData["bgExtract"] = None 

        if len(retData) == 0:
            q2m.put(("finished", retData,), True, None)
            return

        fi = 0
        for idx in range(self.procMod.nF4pp):
            fi += self.procMod.fIntv4pp
            if fi >= nFrames: break
            msg = "Pre-processing ... "
            msg += f'{idx/self.procMod.nF4pp*100:.1f} %'
            q2m.put(("displayMsg", msg), True, None)
            ret = prnt.vRW.getFrame(fi)
            while not ret:
                fi += 1
                ret = prnt.vRW.getFrame(-1)
                if fi >= nFrames: break
            frame = prnt.vRW.currFrame
            
            if "bgExtract" in retData.keys():
            # backgrouind image extraction
                __ = self.procMod.bgSub.apply(frame)

        if "bgExtract" in retData.keys():
            retData["bgExtract"] = True

        # return data 
        q2m.put(("finished", retData,), True, None)

    #---------------------------------------------------------------------------
    
    def setDataCols(self):
        """ Set output data columns depending on animal experiment case 

        Args: None

        Returns:
            dataCols (list): Data column names
            dataInitVal (list): Default value for each data column
            di (dict): Index of each data column
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt

        dataCols = []
        dataInitVal = []
        
        ### Timestamp
        dataCols.append('timestamp')
        dataInitVal.append('')
        
        if prnt.animalECase == "i202408":
            if self.procMod is None: self.procMod = ProcBudgieVideo(self, prnt)
            dataCols, dataInitVal = self.procMod.setDataCols(dataCols, 
                                                             dataInitVal)

        ### store indices of each data column
        di = {} 
        for key in dataCols:
            di[key] = dataCols.index(key)

        return dataCols, dataInitVal, di

    #---------------------------------------------------------------------------
    
    def initAECaseParam(self, aecp):
        """ Set up parameters for the current animal experiment case.
        * Parameter key starts with a letter 'u' means that 
          it will probably modified by users more frequently.

        Args:
            aecp (dict): Parameters for analysis in each experiment

        Returns: 
            aecp (dict): Parameters for analysis in each experiment
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt

        if prnt.animalECase == "i202408":
            if self.procMod is None: self.procMod = ProcBudgieVideo(self, prnt)
            aecp = self.procMod.initAECaseParam(aecp)

        return aecp

    #---------------------------------------------------------------------------
    
    def procImg(self, fImg, tD):
        """ Process frame image to code animal position/direction/behaviour
        
        Args:
            fImg (numpy.ndarray): Frame image array.
            tD (dict): temporary data to process such as 
              hD, bD, hPos, bPos, etc..
        
        Returns:
            fImg (numpy.ndarray): Image to return after processing.
            tD (dict): return data, including hD, bD, hPos, bPos, etc..
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        aec = self.prnt.animalECase
        if aec == "i202408":
            fImg, tD = self.procMod.procImg(fImg, tD)
        else:
            fImg, tD = eval(f'self.proc_{aec}(fImg, tD)')
        return fImg, tD
  
    #---------------------------------------------------------------------------
    
    def drawStatusMsg(self, tD, fImg, fontP={}, pos=(0, 0)):
        """ draw status message and other common inforamtion on fImg
        
        Args:
            tD (dict): Dictionary to retrieve/store calculated data.
            fImg (numpy.ndarray): Frame image.
            fontP (dict): Font parameters. 
            pos (tuple): Position to write.
        
        Returns:
            fImg (numpy.ndarray): Frame image array after drawing. 
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        prnt = self.prnt # parent 

        ### draw file-name & frame index
        if "font" in fontP.keys(): font = fontP["font"]
        else: font = cv2.FONT_HERSHEY_SIMPLEX
        if "scale" in fontP.keys(): scale = fontP["scale"]
        else: scale = 0.5 
        if "thck" in fontP.keys(): thck = fontP["thck"]
        else: thck = 1
        if "fCol" in fontP.keys(): fCol = fontP["fCol"]
        else: fCol = (0, 255, 0)
        fn = path.basename(prnt.inputFP)
        fi = prnt.vRW.fi
        nFrames = prnt.vRW.nFrames-1
        txt = "%s, %i/%i"%(fn, fi, nFrames)
        (tw, th), tBL = cv2.getTextSize(txt, font, scale, thck) 
        cv2.rectangle(fImg, pos, (pos[0]+tw+5,pos[1]+th+tBL), (0,0,0), -1)
        cv2.putText(fImg, txt, (pos[0]+5, pos[1]+th), font, 
                    fontScale=scale, color=fCol, thickness=thck)

        return fImg

    #---------------------------------------------------------------------------
    
    def updatePrevGrayImg(self, flagNone=False):
        """ update previous (preprocessed) gray-image 
        
        Args:
            flagNone (bool)
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.procMod is None:
            if flagNone: self.prevGrayImg = None
            else: self.prevGrayImg = self.grayImg.copy()
        else:
            if flagNone: self.procMod.prevGrayImg = None
            else: self.procMod.prevGrayImg = self.procMod.grayImg.copy()

    #---------------------------------------------------------------------------
    
    def storeFrameImg(self, fImg):
        """ store the frame image of the current frmae of the opened video 
        
        Args: 
            fImg (numpy.ndarray): Frame image.

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt
        inputFP = prnt.vRW.fPath
        ext = "." + inputFP.split(".")[-1]
        folderPath = inputFP.replace(ext, "") + "_frames"
        if not path.isdir(folderPath): # folder doesn't exist
            mkdir(folderPath) # make one 
        fp = path.join(folderPath, "f%07i.jpg"%(prnt.vRW.fi)) # file path
        if not path.isfile(fp): # frame image file doesn't exist
            cv2.imwrite(fp, fImg) # save frame image 

    #---------------------------------------------------------------------------

    def removeDetails(self, img, bgImg):
        """ processing of removing details, then thresholding.

        Args:
            img (numpy.ndarray): image array to process.
            bgImg (numpy.ndarray/ None): background image to subtract.

        Returns:
            diff (numpy.ndarray): grayscale image after BG subtraction.
            edged (numpy.ndarray): grayscale image of edges in 'diff'.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if type(bgImg) == np.ndarray: 
            # get difference between the current frame and the background image 
            diffCol = cv2.absdiff(img, bgImg)
            diff = cv2.cvtColor(diffCol, cv2.COLOR_BGR2GRAY)
        else:
            diffCol = None 
            diff = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        aecp = self.prnt.aecParam
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        if "rdMExOIter" in aecp.keys() and aecp["rdMExOIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_OPEN, 
                        kernel, 
                        iterations=aecp["rdMExOIter"]["value"],
                        ) # to decrease noise & minor features
        if "rdMExCIter" in aecp.keys() and aecp["rdMExCIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_CLOSE, 
                        kernel, 
                        iterations=aecp["rdMExCIter"]["value"],
                        ) # closing small holes
        if "rdThres" in aecp.keys() and aecp["rdThres"]["value"] != -1:
            __, diff = cv2.threshold(
                            diff, 
                            aecp["rdThres"]["value"], 
                            255, 
                            cv2.THRESH_BINARY
                            ) # make the recognized part clear 
        return diffCol, diff
    
    #---------------------------------------------------------------------------
    
    def getEdge(self, grayImg):
        """ Find edges of grayImg using cv2.Canny and cannyTh parameters

        Args:
            grayImg (numpy.ndarray): grayscale image to extract edges.

        Returns:
            (numpy.ndarray): grayscale image with edges.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        return cv2.Canny(grayImg,
                         self.prnt.aecParam["cannyThMin"]["value"],
                         self.prnt.aecParam["cannyThMax"]["value"])

    #---------------------------------------------------------------------------
    
    def drawLocalMaxima(self, gray, threshold=0.2):
        """ calculate distance to background (0) 
        draw only the furthest points far from its background.
        As a result, only center parts of the blobs' thickest parts remain.

        Args:
            gray (numpy.ndarray): grayscale image 
            threshold (float): threshold to determine the peak points

        Returns:
            img (numpy.ndarray): grayscale image after processing
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        # distance transform on the binary image
        distTrans = cv2.distanceTransform(imgClrD, cv2.DIST_L2, 5)
        # normalize the distance transform image for visualization
        distTrans = cv2.normalize(distTrans, None, 0, 1.0, cv2.NORM_MINMAX)
        # threshold the distance transform to get peaks (local maxima)
        _, img = cv2.threshold(distTrans, threshold * distTrans.max(), 255, 0)
        return np.uint8(img)

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass

