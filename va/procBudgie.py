# coding: UTF-8

"""
For processing budgies on videos for VideoAnalysis program (va.py).

Coded and tested on Ubuntu 22.04.

Jinook Oh, Acoustics Research Institute, Wien, Austria.
last edited: 2024-10-28

------------------------------------------------------------------------
Copyright (C) 2024 Jinook Oh
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

import re, subprocess
from time import time, sleep
from datetime import datetime, timedelta
from copy import copy
from os import path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TTFunctional
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as FRCNNv2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as FRCNNv2Wght
from torchvision.models.detection import retinanet_resnet50_fpn_v2 as RNv2 
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights as RNv2Wght
from sklearn.cluster import KMeans

from initVars import *
from modFFC import * 
from modCV import * 
from predWithKF import PredWithKalmanFilter

FLAGS = dict(
                debug = False,
                )

#===============================================================================

class ProcBudgieVideo:
    """ Class for processing a frame image using computer vision algorithms
    to code animal position/direction/behaviour
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent, grandParent):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ##### [begin] setting up attributes -----
        self.prnt = parent
        self.gPrnt = grandParent # VAFrame object in va.py
        
        self.font= cv2.FONT_HERSHEY_SIMPLEX
        self.fScle = parent.fScle # default font scale
        self.fThck = parent.fThck # default font thickness
        self.tW = parent.tW
        self.tH = parent.tH
        self.tBL = parent.tBL

        self.prevGrayImg = None
        ### store the video start time 
        fn = path.basename(grandParent.inputFP)
        # find timestamp (yyyymmdd_hhmmss) in filename
        mtch = re.findall(r"\d{8}_\d{6}", fn)
        if len(mtch) > 0:
            d, t = mtch[0].split("_")
            # set video start time from the timestamp string in filename
            self.videoSTime = datetime(
                    year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]), 
                    hour=int(t[:2]), minute=int(t[2:4]), second=int(t[4:])
                    )
        else:
            # set video start time with the file modification time
            self.videoSTime = getFileMTime(grandParent.inputFP)  

        ### [NOT USING at the moment (2024-09)] for object;bird detection 
        self.odAlg = "" # YOLO, FasterRCNN, RetinaNet 
        if self.odAlg == "":
            self.odMoel = None
        elif self.odAlg == "YOLO":
            self.odModel = torch.hub.load("ultralytics/yolov5", "yolov5s", 
                                          pretrained=True)
            #self.odModel = torch.hub.load("ultralytics/yolov11", "custom", 
            #                              path="custom_yolov11_model.pt")
            self.odModel.conf = 0.2 # detection threshold
            self.odThr = -1
            self.odBirdIdx = -1
        else:
            if self.odAlg == "FasterRCNN":
                wght = FRCNNv2Wght
                # Load the pre-trained model 
                self.odModel = FRCNNv2(weights=wght.DEFAULT)
                self.odThr = 0.3 # detection threshold
            elif self.odAlg == "RetinaNet":
                wght = RNv2Wght 
                # Load the pre-trained model 
                self.odModel = RNv2(weights=wght.DEFAULT)
                self.odThr = 0.2 # detection threshold
            self.odCatNames = wght.DEFAULT.meta["categories"]
            self.odBirdIdx = self.odCatNames.index("bird")
            self.odModel.eval()  # set the model to evaluation mode

        '''
        MOG2: an improved version of the Gaussian Mixture-based 
                Background Segmentation Algorithm (MOG)
        KNN: k-nearest neighbors approach
        GMG: Geospatial Motion Generalization
        '''
        self.bgSubAlg = "KNN" # MOG2, KNN, GMG

        ### budgie info
        self.nSubj = 7 # number of budgies in the aviary
        # n of data (frame) to keep for list data in history 
        self.historyListLen = dict(mar=10, colHist=30*30)
        self.history = {} # History of budgie info; key is an id such as b1. 
          # Value is a dictionary with some info.
          # Data in history is updated in 'updateHistory' function. 
   
        nFrames = self.gPrnt.vRW.nFrames 

        minNF = 20 
        maxNF = 100
        ### set number of frames for pre-processing 
        self.nF4pp = max(minNF, int(nFrames*0.01))
        self.nF4pp = min(self.nF4pp, maxNF, nFrames)
        self.nF4pp = 50 # <<< !!! TEMP.
        self.fIntv4pp = max(1, int(nFrames/self.nF4pp))

        ### init backgroun subtractor object
        bgSub = None 
        if self.bgSubAlg == "MOG2":
            bgSub = cv2.createBackgroundSubtractorMOG2()
            bgSub.setDetectShadows(False)
            bgSub.setVarThreshold(16) # default value is 16
        elif self.bgSubAlg == "KNN":
            bgSub = cv2.createBackgroundSubtractorKNN()
            bgSub.setHistory(500) # 500 by default
            bgSub.setDist2Threshold(400.0) # 400 by default
            bgSub.setDetectShadows(False)
        elif self.bgSubAlg == "GMG":
            bgSub = cv2.bgsegm.createBackgroundSubtractorGMG()
            bgSub.setNumFrames(120)
            bgSub.setDecisionThreshold(0.8) # lower -> more sensitive
        self.bgSub = bgSub
        
        ##### [end] setting up attributes -----

    #---------------------------------------------------------------------------
    
    def setDataCols(self, dataCols, dataInitVal):
        """ Set output data columns 

        Args: 
            dataCols (list): Data column names
            dataInitVal (list): Default value for each data column

        Returns:
            dataCols (list): Data column names
            dataInitVal (list): Default value for each data column
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ''' 4 corners of the detected blobs found with 
        cv2.minAreaRect. Each blob is separted by ampersand and 
        each number is separated by slash. 
        e.g.: x1/y1/x2/y2/x3/y3/x4/y4&x1/y1/x2/y2/ ...
        '''
        #dataCols.append('blobRectPts')
        #dataInitVal.append('')
        
        ### presence of budgie in frame
        dataCols.append('bPresence') 
        dataInitVal.append('False')

        return dataCols, dataInitVal

    #---------------------------------------------------------------------------

    def initAECaseParam(self, aecp):
        """ Set up parameters 

        Args:
            aecp (dict): Parameters for analysis in each experiment

        Returns: 
            aecp (dict): Parameters for analysis in each experiment
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        d = f'Color;HSV threshold for detecting budgie color.'
        vals = {"min": dict(H=20,S=35,V=80), 
                "max": dict(H=140,S=255,V=255)}
        for mm in ["min", "max"]:
            for hsv in ["H", "S", "V"]:
                if hsv == "H": _max = 180
                else: _max = 255
                key = f'uColBudgies-{mm}-{hsv}'
                aecp[key] = dict(value=vals[mm][hsv],
                                 minVal=0, maxVal=_max,
                                 desc=d)
        
        d = "Minimum pixel length to recognize as a blob"
        d += " in detection algorithms."
        key = f'uMinBLen'
        aecp[key] = dict(value=20, minVal=30, maxVal=300, desc=d)

        return aecp

    #---------------------------------------------------------------------------
    
    def procImg(self, fImg, tD):
        """ process a frame in budgie video 
        
        Args:
            fImg (numpy.ndarray): Frame image.
            tD (dict): dictionary to retrieve/store calculated data

        Returns:
            fImg (numpy.ndarray): Return image.
            tD (dict): received 'tD' dictionary, but with calculated data.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        self.fImg = fImg
        prnt = self.prnt # parent
        gPrnt = self.gPrnt # grand parent
        aecp = gPrnt.aecParam # animal experiment case parameters
        fH, fW, fC = fImg.shape # frame shape
        self.frameSz = (fW, fH)
        passedMSec = timedelta(milliseconds=gPrnt.vRW.timestamp)
        # store the frame-timestamp
        tD["timestamp"] = self.videoSTime + passedMSec
        dImgTyp = gPrnt.dispImgType
        # img other than frame image for showing results of 
        # certain detection algorithms
        dImg = np.zeros((fH,fW), dtype=np.uint8) 
        fi = gPrnt.vRW.fi # frame-index
        nFrames = gPrnt.vRW.nFrames # number of frames
        minMCnt = aecp["motionCntThrMin"]["value"] # min. motion contour
        maxMCnt = aecp["motionCntThrMax"]["value"] # max. motion contour
        # min. area for a blob
        minBArea = aecp["uMinBLen"]["value"] ** 2
 
        # get pre-processed gray image
        __, grayImg = prnt.removeDetails(fImg, None)
        self.grayImg = grayImg.copy()
        if dImgTyp == "grayscale" : dImg = grayImg 
        if dImgTyp == "prevgray" : dImg = self.prevGrayImg 

        if self.prevGrayImg is None: # first run
            self.prevGrayImg = grayImg.copy()
            mDiff = np.zeros(grayImg.shape, dtype=np.uint8)
        else:
            # difference image for motion detection
            mDiff = cv2.absdiff(grayImg, self.prevGrayImg)
        
        # apply cv2.threshold function in motion detection
        ret, mDiff = cv2.threshold(mDiff, aecp["motionThr"]["value"], 255, 
                                   cv2.THRESH_BINARY)
        if dImgTyp == "motion": dImg = mDiff  

        ##### [begin] find foreground ----- 
        fgMar = [] # min. area rect of blobs, found with bg-subtraction 
        # apply the background subtractor
        fgImg = self.bgSub.apply(fImg)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        fgImg = cv2.morphologyEx(fgImg, cv2.MORPH_CLOSE, kernel, 
                                 iterations=1)
        fgImgBlob = np.zeros_like(fgImg, dtype=np.uint8)
        # detect blobs (min. area rects) 
        self.detectBlob(fgImg, minBArea, fgImgBlob, False, None) 
        if dImgTyp == "fg": dImg = fgImgBlob 
        ##### [end] find foreground -----

        ##### [begin] find color blob ----- 
        ### get result image, detected with user-defined color
        colThr = {"min":[], "max":[]}
        for mm in ["min", "max"]:
            for hsv in ["H", "S", "V"]:
                key = f'uColBudgies-{mm}-{hsv}'
                colThr[mm].append(aecp[key]['value'])
            colThr[mm] = tuple(colThr[mm])
        blurred = cv2.medianBlur(fImg, 21)
        if dImgTyp == "blur": dImg = blurred 
        fcImg = findColor(blurred, colThr["min"], colThr["max"])  

        fcImgBlob = np.zeros_like(fcImg, dtype=np.uint8)
        # detect blobs (min. area rects) 
        self.detectBlob(fcImg, minBArea, fcImgBlob, False, None)
        if dImgTyp == "color": dImg = fcImgBlob 

        # get the result image of both foreground and color detection results
        fgfcImg = cv2.bitwise_and(fgImgBlob, fcImgBlob)

        fcbwMar = []
        fcbwSPt = []
        if np.sum(fgfcImg) > 0:
            fcbwImg = fgfcImg.copy() 
            
            ### add black/white color around the detected blobs
            ###  (due to black/white feather patterns on budgies' wings)
            cnts, hierarchy = cv2.findContours(fcbwImg, 
                                               cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
            fcBW = np.zeros_like(fcImg, np.uint8)
            for cnt in cnts:
                _mar = cv2.minAreaRect(cnt) 
                box = np.int0(cv2.boxPoints(_mar))
                exPxl = int(np.sqrt(_mar[1][0]**2 + _mar[1][1]**2) * 0.2)
                box = expandBoxPts(box, exPxl=exPxl)
                for c0, c1 in [[(0,0,0), (180,50,150)], 
                               [(0,0,240), (180,30,255)]]:
                    _fc = findColor(fImg, c0, c1, r=box)
                    _fc = cv2.dilate(_fc, kernel, iterations=1)
                    fcBW = cv2.add(fcBW, _fc)
            fcbwImg = cv2.add(fcbwImg, fcBW)
            if dImgTyp == "color2": dImg = fcBW

            # fill holes to get the whole bird area, 
            fcbwImg = cv2.morphologyEx(fcbwImg, cv2.MORPH_CLOSE, kernel, 
                                      iterations=5) # closing small holes
            
            if dImgTyp == "color3": dImg = fcbwImg 

            # get min. area rects and stacked points of budgie-blobs
            fcbwMar, fcbwSPt = self.detectBlob(fcbwImg, minBArea, None, 
                                               True, [])
        ##### [end] find color blob ----- 
 
        w = wx.FindWindowByName("budgieVideoClip_chk", gPrnt.panel["tp"])
        # if this check is True, it means this run is for checking 
        #   budgie-presence to determine where to cut for video clips.
        flagCutVideoClips = widgetValue(w)
        
        bK2disp = []
        if len(fcbwSPt) > 0:
        # if there're valid blobs in the frame 
        
            if not flagCutVideoClips:
                # process the blobs 
                bK2disp = self.procBudgieBlobs(fcbwMar, fcbwSPt, fgfcImg) 

            ### store to tD 
            '''
            key = "blobRectPts"
            for mar in fcbwMar[clrT]:
                box = cv2.boxPoints(mar)
                box = np.int0(box)
                for _i in range(4):
                    if tD[key] != "" and _i == 0: tD[key] += '&'
                    tD[key] += f'{box[_i][0]}/{box[_i][1]}'
                    if _i < 3: tD[key] += "/"
            '''
            
            # store whether budgie is in the frame 
            tD["bPresence"] = 'True'

        if dImgTyp != "frame":
            if len(dImg.shape) < 3:
                # change frame image to the detection image 
                fImg = cv2.cvtColor(dImg, cv2.COLOR_GRAY2BGR) 
            else:
                fImg = dImg

        #### [begin] display info on return image -----  
        # set colors for drawing
        c = dict(motion=(50,255,50), fg=(200,200,0), od=(0,0,255), 
                 budgie=(0,255,255), budgieCol=(50,255,50), 
                 bLbl=(0,0,255))

        ### draw status string
        if fImg.shape[0] < 1000: fontP = dict(scale=0.5, thck=1)
        else: fontP = dict(scale=1.0, thck=2)
        pos = (5, 5)
        fImg = prnt.drawStatusMsg(None, fImg, fontP, pos) 

        ### draw min. area rects of budgies in this frame
        for bK in bK2disp:
            mar = self.history[bK]["mar"][-1]
            box = np.int0(cv2.boxPoints(mar))
            cv2.drawContours(fImg, [box], 0, c["budgie"], 6)
 
        if flagCutVideoClips:
            if eval(tD["bPresence"]):
                cv2.rectangle(fImg, (0,0), (fW,fH), c["budgieCol"], 10)
        
        else:
            ### draw min. area rects of blobs of foreground and color detection
            for mar in fcbwMar:
                box = np.int0(cv2.boxPoints(mar))
                cv2.drawContours(fImg, [box], 0, c["budgieCol"], 2) 
           
            ### draw labels of budgies in this frame
            for bK in bK2disp:
                x, y = self.history[bK]["mar"][-1][0]
                cv2.putText(fImg, bK, (int(x),int(y)), self.font, 
                            fontScale=1.0, color=c["bLbl"], thickness=2)

        #### [end] display info on return image -----

        return fImg, tD


        '''
        ##### [begin] find motion points -----
        mPts = [] # motion-points
        _img = mDiff.copy()
        mPxls = int(np.sum(_img)/255)
        if mPxls <= maxMCnt:
            ### get contours of motion
            mode = cv2.RETR_EXTERNAL 
            method = cv2.CHAIN_APPROX_SIMPLE
            if getOpenCVVersion() >= 4.0:
                cnts, hierarchy = cv2.findContours(_img, mode, method)
            else:
                _, cnts, hierarchy = cv2.findContours(_img, mode, 
                                                      method)
            for cnti in range(len(cnts)):
                mr = cv2.boundingRect(cnts[cnti])
                # ignore too small contour
                if mr[2]+mr[3] < minMCnt: continue 
                x = mr[0]+int(mr[2]/2)
                y = mr[1]+int(mr[3]/2)
                # store for drawing on return image later
                mPts.append((x,y)) 
        ##### [end] find motion points -----  
        '''

        '''
        #-----------------------------------------------------------------------
        def calcOpticalFlowInBlobs(grayImg, prevGrayImg, blobSPt):
            p = dict(flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = detectOpticalFlowFarneback(grayImg, prevGrayImg, p) 

            img = np.zeros((fH,fW,3), dtype=np.uint8)
            for spt in blobSPt:
                bMask = self.getBlobMaskImgFromPoints(spt)

                # optical flow vectors for the blob's pixels
                flowInBlob = flow[bMask>0]

                # get the coordinates of the blob's pixels
                blobCoords = np.argwhere(bMask > 0)

                # calculate magnitude and angle of the flow vectors
                mag, ang = cv2.cartToPolar(flowInBlob[..., 0], 
                                           flowInBlob[..., 1])

                # map angle (in radians) to degrees for hue (0-179; OpenCV HSV)
                hue = ang * 180 / np.pi / 2
                # normalize magnitude for value  
                value = np.clip((mag * 255 / np.max(mag)), 0, 255)
                ### assign hue and value to HSV image for the blob's pixels
                for i, (y, x) in enumerate(blobCoords):
                    img[y, x, 0] = hue[i] # hue (direction)
                    img[y, x, 1] = 255 # saturation 
                    img[y, x, 2] = value[i] # value (magnitude)
            
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            return img

        #-----------------------------------------------------------------------
        if dImgTyp == "color2":
            dImg = calcOpticalFlowInBlobs(grayImg, self.prevGrayImg, fcbwSPt)
        '''

    #---------------------------------------------------------------------------
   
    def detectBlob(self, img, minBArea, dImg, flagMar=True, sPtLst=None):
        """ Detect blobs using connectedComponentsWithStats and 
        return min. area rects (cv2.boxPoints)

        Args:
            img (numpy.ndarray): grayscale image to detect blobs
            minBArea (int): minimum area to recognize as a blob
            dImg (numpy.ndarray): display-image to display blobs only
            flagMar (bool): whether to calculate min. area rect
            sPtLst (list): to store stacked points of blobs
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
            
        mar = [] # min. area rect to return

        # find blobs in the color-detection result image
        #   using connectedComponentsWithStats
        ccOutput = cv2.connectedComponentsWithStats(img, 
                                                    connectivity=4)
        nLabels = ccOutput[0] # number of labels
        labeledImg = ccOutput[1]
        # stats = [left, top, width, height, area]
        stats = list(ccOutput[2])
        for li in range(1, nLabels):
            l, t, w, h, a = stats[li]

            if a < minBArea: continue # this contour is too small. ignore.

            if dImg is not None: dImg[labeledImg==li] = 255 # draw the blob

            if flagMar or sPtLst is not None:
                ### calculate minimum area rect for this blob 
                ptL = np.where(labeledImg==li) 
                sPt = np.hstack((
                          ptL[1].reshape((ptL[1].shape[0],1)),
                          ptL[0].reshape((ptL[0].shape[0],1))
                          )) # stacked points
            if sPtLst is not None:
                sPtLst.append(sPt)

            if flagMar:
                mar.append(cv2.minAreaRect(sPt)) 
        return mar, sPtLst

    #---------------------------------------------------------------------------
    
    def getBlobMaskImgFromPoints(self, sPt): 
        """ make a mask image with the stacked points of blobs
        
        Args:
            sPt (numpy.ndarray): numpy array of stacked points
        
        Returns:
            bMask (numpy.ndarray): return image
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        fW, fH = self.frameSz
        bMask = np.zeros((fH, fW), dtype=np.uint8)
        bMask[sPt[:, 1], sPt[:, 0]] = 255
        return bMask

    #---------------------------------------------------------------------------
    
    def procBudgieBlobs(self, fcbwMar, fcbwSPt, fgfcImg):
        """ process blobs in the current frame with the blob info in history.
        
        Args:
            fcbwMar (list): list of min. area rects of results of 
                            both foreground and color detection
            fcbwSPt (list): list of numpy stacked points of results of 
                            both foreground and color detection
            fgfcImg (numpy.ndarray): Image of blobs both in foreground and 
                                     color detection
        
        Returns:
            bK2disp (list): list of budgie-keys to display on output image 
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        _debug = True
        fi = self.gPrnt.vRW.fi
        fW, fH = self.frameSz
        fImg = self.fImg
        bH = self.history # budgie info history
        if _debug:
            print("---------------------------------------\n")
            print(f"Frame-idx: {fi},  'procWithHistory' ---")

        bK2disp = []
        bKeys = list(bH.keys()) # budgie key list
        if len(bH) == 0: flag1stData = True
        else: flag1stData = False 
        thr = {} # thresholds
        
        ''' Distance from a color-blob to a budgie (history), 
        in terms of ratio of distance to the diagonal-length of the budgie.
        [0] to determine whether it's accurately matching with the prediction
        [1] normal threshold, [2] threshold for fast moving/flying budgie
        '''
        thr["dist"] = [0.1, 0.4, 1.2]
        
        ''' Distance from a color-blob to another color-blob to consider it 
        as a part of one budgie; threshold = thr["distB2b"] * sum(dL-list)'''
        thr["distB2b"] = 0.6
        
        ''' Minimum color similarity to consider as a part of a budgie'''
        thr["minClr"] = 0.5
        
        ''' Color similarity to consider as the same budgie'''
        thr["sameBClr"] = 0.95
        
        ''' area ratio of a a color-blob in current frame to 
        a budgie in the prev. frame
        [0]: Too small if it's below this.
        [1]: If it's in between [0] and [1]. Maybe a small section
            of its body color was incorrectly detected in this frame.
        [2]: It's in normal range if it's between [2] and [3].
        [3]: Too large if it's over this. Maybe two budgies in the prev. frame
            got close to each other, forming one big color-blob.'''
        thr["aR"] = [0.25, 0.5, 0.8, 1.3]
        
        edges = [ ((0,0),  (0,fH)), 
                  ((0,0),  (fW,0)), 
                  ((fW,0), (fW,fH)), 
                  ((0,fH), (fW,fH)) ] # edges of the scene 

        ### get some color-blob data
        cbData = [] 
        for cbi, cSPt in enumerate(fcbwSPt): 
            if _debug:
                mar = fcbwMar[cbi]
                (x, y), (w, h), _ = mar
                x, y = (int(x), int(y))
                w, h = (int(w), int(h))
                txt = f"  color-blob [{cbi}] pos:{x, y}, size:{w, h}" 
                print(txt)
            bMask = self.getBlobMaskImgFromPoints(cSPt)
            cbData.append(dict(ch = getColorHistogram(fImg, bMask)))

        ### prep. budgie data in history
        pbData = {}
        for bK in copy(bKeys):
            ### get predicted position & size
            kfP = bH[bK]["kfP"]
            if kfP is None:
                # copy x, y, w, h from the last MAR
                (hx,hy), (hw,hh), _ = bH[bK]["mar"][-1]
            else:
                # x & y of budgie, predicted from history
                hx, hy = kfP.getPred()
                # width & height of budgie, predicted from history
                hw, hh = bH[bK]["kfS"].getPred()
                #cv2.circle(fImg, (int(hx),int(hy)), 5, (0,0,255), -1)
            pbData[bK] = dict(pos=(hx, hy),
                              dL=np.sqrt(hw**2 + hh**2),
                              area=bH[bK]["prevSPt"].shape[0])

        def bIsClose2cb(pbData, cbMar, distThr):
            ''' function to determine whether budgie's predicted position 
            (Kalman-filter prediction) is close enough to the color blob'''
            bx, by = pbData["pos"]
            (cbx, cby), (cbw, cbh), _ = cbMar
            dist = np.sqrt((bx-cbx)**2 + (by-cby)**2)
            distR = dist / pbData["dL"] 
            if _debug: print(f"\t  dist rat.: {distR:.3f}, thr:{distThr}")
            if distR < distThr: return True
            else: return False

        def isBlobFarFromEdge(cbx, cby, cbDL):
            ''' function to determine whether a blob is far from the edge 
            of the scene'''
            if (fW*0.25 < cbx < fW*0.75) and (fH*0.25 < cby < fH*0.75):
            # in the middle of the scene
                return True 
            dists = [cbx, cby, fW-cbx, fH-cby]
            edge = edges[dists.index(min(dists))]
            d2edge = calc_pt_line_dist((cbx,cby), edge)
            if d2edge > cbDL*0.75: return True
            else: return False 

        assignedCBI = [] # assigned color-blob index
        ##### [begin] process color-blobs with budgie data in history ----- 
        _bKeys = copy(bKeys)

        ### Update the budgie history when the blob is quite close 
        ###   to the prediction in its distance, color and size ratio.
        for bi, bK in enumerate(_bKeys):
        # go through tracked budgies in the history
            isFound = False
            txt = ""
            for cbi, cSPt in enumerate(fcbwSPt):
            # go through each color blob
                if cbi in assignedCBI: continue
                # color-similarity
                cs = self.comp2MedClr(cbData[cbi]["ch"], bH[bK]["colHist"])
                # size (area) ratio; this blob to the data in history
                aR = cSPt.shape[0] / pbData[bK]["area"]
                # very similar color 
                if cs > thr["sameBClr"]:
                    if thr["aR"][2] <= aR < thr["aR"][3]:
                    # similar size
                        isFound = True
                        txt += f"  * [clr({cs:.3f}, {thr['sameBClr']})"
                        txt += f", sz({aR:.3f}, {thr['aR']})] "
                        break
                if bIsClose2cb(pbData[bK], fcbwMar[cbi], thr["dist"][0]):
                # distance is very close 
                    cs = self.comp2MedClr(cbData[cbi]["ch"], bH[bK]["colHist"])
                    if cs > thr["minClr"]:
                    # color is similar enough
                        if thr["aR"][2] <= aR < thr["aR"][3]:
                        # size ratio is acceptable
                            isFound = True
                            txt += "  * [dist"
                            txt += f",  clr({cs:.3f}, {thr['sameBClr']})"
                            txt += f", sz({aR:.3f}, {thr['aR']})] "
                            break
            if isFound:
                # add this blob as a budgie in the history
                self.updateHistory(bK, fcbwMar[cbi], cbData[cbi]["ch"], cSPt) 
                bK2disp.append(bK)
                assignedCBI.append(cbi)
                _bKeys[bi] = None # remove this budgie key
                if _debug:
                    txt += f"blob-{cbi:02d} -> budgie-{bK}."
                    print(txt)

        for bK in _bKeys:
        # go through tracked budgies in the history
            if bK is None: continue
            if _debug:
                txt = f"  Procssing {bK} "
                if bH[bK]["isFlying"]: txt += "[flying] "
                print(txt + " ...")

            cbi2a = [] # color-blob index to assign to this budgie (bK) 

            if bH[bK]["isFlying"]: distThr = thr["dist"][2]
            else: distThr = thr["dist"][1]

            ### get color-blob indices that are close enough to 
            ###   this budgie's expected position in this frame 
            for cbi, cSPt in enumerate(fcbwSPt):
            # go through each color blob
                if cbi in assignedCBI: continue
                if _debug: print(f"\tcomparing with blob-{cbi:02d} ..")
                if bIsClose2cb(pbData[bK], fcbwMar[cbi], distThr):
                # close distance
                    cbi2a.append(cbi) # append this blob-index
            if _debug: print(f"\tclose blob idx: {str(cbi2a)}") 

            nCB = len(cbi2a) # n of color-blobs close enough to this budgie
            if nCB == 0 : # no close color-blob
                continue # to the next budgie
            elif nCB == 1: # single color-blob
                cbi = cbi2a[0]
                sPt = fcbwSPt[cbi]
                ch = cbData[cbi]["ch"]
                mar = fcbwMar[cbi]
            elif nCB > 1: # multiple color-blobs
                ### combine stacked-points of the color-blobs 
                ###   and re-calculate MAR and CH with it
                cSPtLst = [fcbwSPt[cbi] for cbi in cbi2a]
                sPt = np.vstack(cSPtLst)
                bMask = self.getBlobMaskImgFromPoints(sPt)
                ch = getColorHistogram(fImg, bMask)
                mar = cv2.minAreaRect(sPt)
 
            ##### [begin] consider size & color difference -----

            # size difference
            aR = sPt.shape[0] / pbData[bK]["area"]
            
            # color-similarity
            cs = self.comp2MedClr(ch, bH[bK]["colHist"])

            if _debug:
                print(f"\tarea rat.: {aR:.3f}, thr:{thr['aR']}")
                print(f"\tcolor-sim.: {cs:.3f}, thr:{thr['minClr']}")

            flagUpdateHistory = False
            if aR < thr["aR"][0]:
            # became too small
                cbi2a = []
            
            elif thr["aR"][0] < aR < thr["aR"][1]:
            # became rather small (maybe incorrect color-detetion)
                isFlyingAtEdge = False # budgie flying at the edge of the scene
                if bH[bK]["isFlying"]: # this budgie is flying
                    bx, by = pbData[bK]["pos"]
                    if not isBlobFarFromEdge(bx, by, pbData[bK]["dL"]):
                        isFlyingAtEdge = True 

                if not isFlyingAtEdge and cs > thr["minClr"]:
                # if color is similar enough,
                    ### copy the previous frame data
                    # copy size of the budgie from the prev. frame
                    mar = (mar[0], bH[bK]["mar"][-1][1], mar[2])
                    ch = copy(bH[bK]["colHist"][-1])
                    sPt = copy(bH[bK]["prevSPt"])
                    flagUpdateHistory = True

            elif aR > thr["aR"][3]:
            # The size became too big. Maybe two or more budgies 
            #   in the prev. frame got too close to each other forming 
            #   a large color-blob in this frame
                bK2s = [bK]
                for _bi, _bK in enumerate(_bKeys):
                    if _bK == bK or _bK is None: continue
                    if bH[_bK]["isFlying"]: _distThr = thr["dist"][2]
                    else: _distThr = thr["dist"][1] * aR 
                    isClose = bIsClose2cb(pbData[_bK], mar, _distThr)
                    if _debug: print(f"\tisClose with {_bK}: {isClose}")
                    if isClose: # in close distance 
                        bK2s.append(_bK) # append this budgie key
                        _bKeys[_bi] = None
                if len(bK2s) > 1:
                    # store budgie-key to display
                    bK2disp += bK2s
                    # split this color-blob (history will be updated)
                    self.splitBlobWithBudgies(bK2s, sPt)
                else: 
                    if bH[bK]["isFlying"]:
                    # the size difference is big, but maybe it's due to the
                    #   wings while flying.
                        flagUpdateHistory = True # update
                    else:
                        ### copy the previous frame data
                        # copy size of the budgie from the prev. frame
                        mar = (mar[0], bH[bK]["mar"][-1][1], mar[2])
                        ch = copy(bH[bK]["colHist"][-1])
                        sPt = copy(bH[bK]["prevSPt"])
                        flagUpdateHistory = True # update
                        #cbi2a = [] # ignore this color-blob for this budgie

            else:
            # in acceptable range
                if cs > thr["minClr"]:
                # if color is similar enough,
                    flagUpdateHistory = True
            ##### [end] consider size & color difference -----
            
            if flagUpdateHistory:
                self.updateHistory(bK, mar, ch, sPt)
                bK2disp.append(bK) 

            if cbi2a != []: assignedCBI += cbi2a
        ##### [end] process blobs with budgie data in history ----- 

        if len(assignedCBI) < len(fcbwSPt):
        # there're left-over color-blobs 
            # color-blob indices, which are not assigned yet
            naCBI = list(set(range(len(fcbwSPt))) - set(assignedCBI))

            ##### [begin] group blobs -----
            gCBIL = []
            for cbi in naCBI:
            # go through non-assigned color-blob
                (x1,y1), (w1,h1), _ = fcbwMar[cbi]
                ch1 = cbData[cbi]["ch"]
                dL1 = np.sqrt(w1**2 + h1**2)
                
                isAdded2group = False
                for group in gCBIL:
                # go through already existing groups
                    distLst = []
                    dLLst = [dL1]
                    csLst = []
                    for _cbi in group:
                        (x2,y2), (w2,h2), _ = fcbwMar[_cbi]
                        ch2 = cbData[_cbi]["ch"]
                        dL2 = np.sqrt(w2**2 + h2**2)
                        dLLst.append(dL2)
                        distLst.append(np.sqrt((x1-x2)**2 + (y1-y2)**2))
                        cs = cv2.compareHist(ch1, ch2, cv2.HISTCMP_CORREL)
                        csLst.append(cs)
                    if max(csLst) > thr["minClr"] and \
                      max(distLst) < sum(dLLst)*thr["distB2b"]:
                    # color is similar enough and distance is close enough
                        group.append(cbi)
                        isAdded2group = True
                        break
                if not isAdded2group: 
                    gCBIL.append([cbi])
                    continue
            ##### [end] group blobs -----

            gcbData = []
            for group in gCBIL:
                gSPt = [fcbwSPt[cbi] for cbi in group]
                gSPt = np.vstack(gSPt)
                mar = cv2.minAreaRect(gSPt)
                bMask = self.getBlobMaskImgFromPoints(gSPt)
                ch = getColorHistogram(fImg, bMask)
                gcbData.append((mar, ch, gSPt))
            
            if flag1stData: # no history data available
                ### register blobs as new budgies and return
                for mar, ch, sPt in gcbData:
                    bK = f'b{len(bH)+1:02d}'
                    self.updateHistory(bK, mar, ch, sPt)
                    bK2disp.append(bK)
                return bK2disp 

            ##### [begin] group non-assigned color-blobs -----  
            for mar, ch, sPt in gcbData:
            # go through non-assigned color-blob after grouping

                (cbx, cby), (cbw, cbh), _ = mar
                cbDL = np.sqrt(cbw**2 + cbh**2)
                if isBlobFarFromEdge(cbx, cby, cbDL):
                # too far from edge of the scene.
                # discard this color blob; budgie can't pop up in the middle
                    continue

                ### if this blob doesn't overlap with fg-color image at all,
                ###   ignore it.
                b = self.getBlobMaskImgFromPoints(sPt)
                b = cv2.bitwise_and(fgfcImg, b)
                if np.sum(b) == 0:
                    continue

                ### register as a new budgie entered to the scene 
                bK = f'b{len(bH)+1:02d}'
                self.updateHistory(bK, mar, ch, sPt)
                bK2disp.append(bK)
            ##### [end] group non-assigned color-blobs -----

        if _debug:
            txt = f"*** budgies in the scene: {str(bK2disp)}\n"
            txt += "---------------------------------------\n"
            print(txt)

        for bK in set(bKeys) - set(bK2disp):
        # go through budgie keys in history, which didn't appear in this frame
            ### just update the position with the KF prediction
            prevMar = bH[bK]["mar"][-1]
            mar = (pbData[bK]["pos"], prevMar[1], prevMar[2])
            self.updateHistory(bK, mar, None, None)

        return bK2disp

    #---------------------------------------------------------------------------
    
    def updateHistory(self, bK, mar, ch, sPt):
        """ Updating the budgie data history
        
        Args:
            bK (str): budgie key 
            mar (tuple): result of cv2.minAreaRect
            ch (numpy.ndarray): result of cv2.calcHist
            sPt (numpy.ndarray): numpy stacked points of the blob (budgie)
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        fi = self.gPrnt.vRW.fi 
        
        _debug = True
        if _debug:
            print(f"  beginning of 'updateHistory' ---")

        if bK not in self.history.keys():
            # init.
            self.history[bK] = dict(
                                        mar=[], # min. area rect
                                        colHist=[], # color histogram
                                        prevSPt=None, # previous points 
                                        isFlying=True,
                                        updatedFI=-1, # last updated frame-idx
                                        kfP=None, # kalman-filter for position
                                        kfS=None, # kalman-filter for size 
                                        )
        
        history = self.history[bK] # history of this budgie
    
        if len(history["mar"]) > 1 and sPt is not None:
            ### determine whether this budgie is flying
            if _debug: txt = f"  {bK},"
            if history["updatedFI"] < fi-2: # lost tracking
                history["isFlying"] = False
                if _debug: txt += " (lost track)"
            else:
                px, py = history["mar"][-1][0] # (x,y) in prev. frame
                (x, y), (w, h), _ = mar
                dist = np.sqrt((x-px)**2 + (y-py)**2)
                diagL = np.sqrt(w**2 + h**2)
                spd = dist/ diagL
                if spd > 0.2: history["isFlying"] = True
                else: history["isFlying"] = False
                if _debug:
                    txt += f" dist:{dist:.3f}, diagL:{diagL:.3f},"
                    txt += f" spd: {spd:.3f},"
            if _debug:
                txt += f" isFlying: {history['isFlying']}"
                print(txt)

        history["mar"].append(mar) # update min. area rect
        ### update color-histogram & stacked points
        if sPt is None: # budgie was not seen in this frame
            history["colHist"].append(history["colHist"][-1])
            # [[ NOT implemented ]] update positions of sPt?
        else:
            history["colHist"].append(ch)
            history["prevSPt"] = sPt
        
        ### update kalman-filter
        marLen = len(history["mar"])
        if marLen == 2 or \
          (sPt is not None and history["updatedFI"] < fi-2 and marLen > 1):
        # initial state or track was lost
            ### init. kalman-filter
            for lbl in ["P", "S"]:
                key = f"kf{lbl}"
                if lbl == "P": # position
                    params = dict(mErr=10, mUncert=0.4, 
                                  initSpat=100, initDynamic=200)
                    dI = 0
                elif lbl == "S": # size 
                    params = dict(mErr=50, mUncert=0.3, 
                                  initSpat=100, initDynamic=200)
                    dI = 1
                history[key] = PredWithKalmanFilter(params)
                m = history["mar"]
                history[key].initState(m[-2][dI], m[-1][dI])
        else:
            if history["kfP"] is not None:
                pos, sz, ang = history["mar"][-1]
                history["kfP"].update(pos)
                history["kfS"].update(sz)
      
        ### remove data when there're too many 
        for key in ["mar", "colHist"]:
            if len(history[key]) > self.historyListLen[key]:
                history[key].pop(0)

        if sPt is not None: # if budgie was seen in this frame
            history["updatedFI"] = copy(fi) # store last updated frame index
        
        if _debug: print("  end of 'updateHistory' --- \n")

    #---------------------------------------------------------------------------
    
    def splitBlobWithBudgies(self, bK2s, currSPt):
        """ split one blob joint by multiple budgie blobs
        
        Args:
            bK2s (list): List of budgie keys inside this blob; currSPt
            currSPt (numpy.ndarray): numpy stacked points of the blob
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        st = time()

        fW, fH = self.frameSz
        bMask = self.getBlobMaskImgFromPoints(currSPt)

        # get all non-zero pixel coordinates
        coord0 = np.column_stack(np.where(bMask > 0))

        # apply K-means clustering
        km = KMeans(n_clusters=len(bK2s), random_state=123)
        km.fit(coord0)

        # Get the labels assigned to each pixel
        labels = km.labels_

        outputImg = np.zeros((fH,fW,3), dtype=np.uint8) 
        for i, label in enumerate(np.unique(labels)):
            indices = np.where(labels == label)[0]
            coord1 = coord0[indices]
            ci = coord1[:,1]
            ri = coord1[:,0]
            outputImg[ri,ci,:] = self.prnt.cluster_cols[i] 

            # stacked points
            sPt = np.hstack((
                    ci.reshape((ci.shape[0], 1)), # x
                    ri.reshape((ri.shape[0], 1)) # y
                    ))
            # min. area rect
            mar = cv2.minAreaRect(sPt)
            bMask = self.getBlobMaskImgFromPoints(sPt)
            # color histogram
            ch = getColorHistogram(self.fImg, bMask)

            ### get the closest budgie's key index
            (x, y), _, _ = mar
            dists = []
            for bK in bK2s:
                (px, py), _, _ = self.history[bK]["mar"][-1]
                dists.append(np.sqrt((px-x)**2 + (py-y)**2))
            mdi = dists.index(min(dists))

            # store info in the history 
            self.updateHistory(bK2s[mdi], mar, ch, sPt)

            bK2s.pop(mdi) # remove the already assigned key

        #cv2.imwrite("x.jpg", outputImg)

    #---------------------------------------------------------------------------
    
    def comp2MedClr(self, ch, chLst):
        ''' compare color-historygram; cs to median histogram of list of 
        color-histograms; chLst.

        Args:
            ch (numpy.ndarray): color-histogram 
            chLst (list): List of color-histograms

        Returns:
            (flost): Result value of cv2.compareHist
        '''
        stCH = np.stack(chLst, axis=0)
        medCH = np.median(stCH, axis=0)
        medCH = cv2.normalize(medCH, medCH)
        return cv2.compareHist(ch, medCH, cv2.HISTCMP_CORREL)

    '''
    #---------------------------------------------------------------------------
    
    def _splitBlobWithBudgies(self, bK2s, currSPt):
        """ 
        
        Args:
            bK2s (list): List of budgie keys inside this blob; currSPt
            currSPt (numpy.ndarray): numpy stacked points of the blob
        
        Returns:
            bK2disp (list): List of budgie keys to add to bK2disp
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        st = time()
        bK2disp = [] # budgie-key to display

        fW, fH = self.frameSz

        ### get list of contours of budgie blobs in the previous frame
        prevCnts = []
        for key in bK2s:
            bm = self.getBlobMaskImgFromPoints(self.history[key]["prevSPt"])
            cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
            prevCnts.append(cnts)

        # blob mask image of the current frame
        currBM = self.getBlobMaskImgFromPoints(currSPt)

        # get optical flow
        flow = detectOpticalFlowFarneback(self.prevGrayImg, self.grayImg)

        # initialize markers
        markers = np.zeros_like(currBM, dtype=np.int32)

        idx = 1
        for cnts in prevCnts:
            for contour in cnts:
                pts = contour[:, 0] # get all points in the contour
                # compute the mean optical flow vector for the contour
                flowVector = flow[pts[:, 1], pts[:, 0]]
                # get the median flow vector (dx, dy)
                medFlow = np.median(flowVector, axis=0)
                # apply the median flow vector to each point
                newContour = pts + medFlow
                newContour = newContour.astype(int)
                # ensure newX and newY are in valid range
                newContour = np.clip(newContour, [0, 0], [fW, fH])
                # assign a marker with a unique label for each blob
                markers[newContour[:, 1], newContour[:, 0]] = copy(idx)
            idx += 1

        ### Assign index to the non-assigned blob pixels in the current frame
        # set to zero where there's nothing in the current scene
        markers[currBM == 0] = 0
        # get leftover pixels in the current frame
        leftPix = np.column_stack(np.where((currBM == 255) & (markers == 0)))
        # non-zero markers
        nonZeroMIdx = np.unique(markers[markers > 0])
        # calculate centroid of each blob
        blobCent = np.array([
            np.mean(
                    np.column_stack(np.where(markers == idx)), 
                    axis=0
                    )
            for idx in nonZeroMIdx 
        ])
        if leftPix.size > 0 and blobCent.size > 0:
            # calculate distances from leftover pixels to blob centroids
            distances = np.linalg.norm(leftPix[:, None, :] - \
                                       blobCent[None, :, :], axis=2)
            # find the closest blob for each leftover pixel
            closeIdx = np.argmin(distances, axis=1)
            # assign the closest blob index to markers
            markers[tuple(leftPix.T)] = nonZeroMIdx[closeIdx]

        outputImg = np.zeros((fH,fW,3), dtype=np.uint8) 
        for i, mIdx in enumerate(nonZeroMIdx):

            outputImg[markers==mIdx] = self.prnt.cluster_cols[i] 
            
            # find the coordinates of the marker
            coords = np.column_stack(np.where(markers==mIdx))
            
            # stacked points
            sPt = np.hstack((
                    coords[:, 1].reshape((coords[:, 1].shape[0], 1)), # x
                    coords[:, 0].reshape((coords[:, 0].shape[0], 1)) # y
                    ))
            # min. area rect
            mar = cv2.minAreaRect(sPt)
            bMask = self.getBlobMaskImgFromPoints(sPt)
            # color histogram
            ch = getColorHistogram(self.fImg, bMask)

            ### get the closest budgie's key index
            if len(bK2s) == 1:
                mdi = 0
            else:
                (x, y), _, _ = mar
                dists = []
                for bK in bK2s:
                    (px, py), _, _ = self.history[bK]["mar"][-1]
                    dists.append(np.sqrt((px-x)**2 + (py-y)**2))
                mdi = dists.index(min(dists))

            bK2disp.append(bK2s[mdi]) # store budgie-key to display

            # store info in the history 
            self.updateHistory(bK2s[mdi], mar, ch, sPt)

            bK2s.pop(mdi) # remove the already assigned key

        print(time()-st)

        cv2.imwrite("x.jpg", outputImg)

        return bK2disp

    '''
    #---------------------------------------------------------------------------
    
    def extractVideoClips(self, oData):
        """ extract video clips from the original video file 
        
        Args:
            oData (list): Output data for each frame
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.prnt
        gPrnt = self.gPrnt
     
        def getDur(sIdx, eIdx, fIntv):
            #sIdx = max(0, sIdx-30)
            eIdx -= nTol
            startT = timedelta(seconds = sIdx*fIntv)
            currT = timedelta(seconds = eIdx*fIntv)
            dur = currT - startT
            return startT, dur
        
        fps = gPrnt.vRW.fps
        nTol = fps * 3 # n of frames to tolerate with budgie absence 
        cntAbsence = 0 # n of frames of continous budgie absence 
        fIntv = 1.0 / fps # frame interval 
        timeLst = [] # list of tuples; 
                     #   (start-time, duration) for ffmpeg command
        sIdx = -1 # start-index
        bpIdx = gPrnt.di["bPresence"]
        totalDur = timedelta(seconds=0) 
        for fi in range(len(oData)):
            if eval(oData[fi][bpIdx]):
            # budgie presence
                if sIdx == -1: sIdx = int(fi)
                cntAbsence = 0
            else:
            # budgie absence 
                if sIdx != -1:
                # start-index is available 
                    cntAbsence += 1 # count the budgie absence
                    if cntAbsence > nTol:
                    # too many absence frames
                        ### store the end time 
                        startT, dur = getDur(sIdx, fi, fIntv)
                        if dur > timedelta(seconds=1):
                            totalDur += dur
                            timeLst.append((sIdx, fi, str(startT), str(dur)))
                        cntAbsence = 0
                        sIdx = -1 
        if sIdx != -1:
            startT, dur = getDur(sIdx, fi, fIntv)
            totalDur += dur
            timeLst.append((sIdx, fi, str(startT), str(dur))) 

        def runFFMpegCmd(q2m, timeLst, inputFP, outputFolder):
            ### get budgie name
            bName = inputFP.split(".")[0].split("_")[-1]
            try: bName = int(bName)
            except: pass
            if type(bName) == int: bName = None

            for i, (sIdx, eIdx, sTime, dur) in enumerate(timeLst):

                st = int(sTime.replace(":","").split(".")[0])
                st = f"{st:06d}"
                if bName is None: fn = f"clip_{i:03d}_{st}.mp4"
                else: fn = f"{bName}_{i:03d}_{st}.mp4"
                outputFP = path.join(outputFolder, fn)
                
                cmd = "ffmpeg"
                #cmd += f" -i '{inputFP}' -ss {sTime} -t {dur}"
                cmd += f" -ss {sTime} -i '{inputFP}' -t {dur}"
                cmd += f" -c:v libx264 -crf 17 -pix_fmt yuv420p"
                cmd += f" -y '{outputFP}'"
                cmd = cmd.split(" ")
                print(f"[{i+1}/{len(timeLst)}]\n", cmd)
                
                process = subprocess.Popen(cmd, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.STDOUT, 
                                           universal_newlines=True,
                                           bufsize=1)
                try:
                    for line in process.stdout:
                        q2m.put(("displayMsg", line), True, None)
                finally:
                    process.stdout.close()
                    process.wait()
                sleep(0.1)
            
            q2m.put(("displayMsg", "Finished FFMpeg commands."), True, None)
            q2m.put(("finished",), True, None)
            txt = f"{len(timeLst)} clips were extracted.\n"
            txt += f"Total duration: {str(totalDur)}"
            print(txt)

        args = (gPrnt.q2m, timeLst, gPrnt.inputFP, gPrnt.outputFP,)
        startTaskThread(gPrnt, "ffmpegCmd", runFFMpegCmd, args=args)

    #---------------------------------------------------------------------------
#===============================================================================

if __name__ == '__main__':
    pass



