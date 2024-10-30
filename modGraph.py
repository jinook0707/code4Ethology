"""
Functions for antSim

last edited: 2023-09-08
"""

import sys
from os import path
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal

from modFFC import *
from modCV import *

MyLogger = setMyLogger("antSimFuncs")
DEBUG = False

#-------------------------------------------------------------------------------

def setColors():
    """ set colors for graphs 

    Args: None

    Returns: None
    """
    if DEBUG: logging.info(str(locals()))

    ### colors (RGB) for matplotlib graphs
    c = dict(
                dLn=(0.2, 0.2, 0.7), # data line 
                rLn0=(1.0, 0.5, 0.0), # regression line
                rLn1=(0.2, 0.8, 0.2), # 2nd regression line
                flier=(1.0, 0.0, 0.0),
                medianLn=(0.0, 0.0, 0.0), # median line in box plot 
                NM = (0.25, 0.25, 0.4), # number of motions in mComp graph
                MP = (1.0, 0.0, 0.0) # mean power in mComp graph
                )
    ### colors (BGR) for cv2 drawing functions on an image array 
    c["inten"] = (100, 100, 100) # intensity bar
    c["cvFont"] = (0, 0, 0) # font color for cv2.putText
    c["outputFr"] = (0, 0, 0) # frame of an output in intensity graph
    c["outputFrFinished"] = (0, 0, 0) # frame of an output in intensity graph
    # line color for indicating longer inactivity duration boost
    c["lidb"] = (255,75,75) 
    return c

#-------------------------------------------------------------------------------

def setPLTParams(dpi=200, nRow=2, nCol=3, fSz=8, lW=1, spSz=(2.0,1.25)):
    """ set some parameters for matplotlib figure 

    Args:
        dpi (int): 
        nRow (int): # of rows.
        nCol (int): # of columns.
        fSz (int): font size.
        lW (int): line width.
        spSz (tuple): size for each subplot

    Returns:
        None
    """
    if DEBUG: logging.info(str(locals()))
    
    plt.rcParams["figure.dpi"] = dpi 
    plt.rcParams["savefig.dpi"] = dpi 
    figW = spSz[0]*nCol
    figH = spSz[1]*nRow
    plt.rcParams["figure.figsize"] = (figW, figH) 
    plt.rcParams["font.size"] = fSz 
    plt.rcParams["lines.linewidth"] = lW 

#-------------------------------------------------------------------------------

def setYFormat(maxY, subP=None, formatStr="{:,.0f}", flagShowUnit=True):
    """ Set format on the y-axis values

    Args:
        maxY (int): Max value
        subP (None/matplotlib.pyplot.AxesSubplot)
        formatStr (str): Format string
        flagShowUnit (bool): Whether to display unit next to the value

    Returns:
        dvsrStr (str) 
    """
    if DEBUG: logging.info(str(locals()))

    maxY = int(maxY)
    length = len(str(maxY))
    dvsr = 1
    unit = ""
    if length > 9:
        dvsr = 1e9
        unit = "B"
    elif length > 6:
        dvsr = 1e6
        unit = "M"
    elif length > 3:
        dvsr = 1e3
        unit = "K"
    if not flagShowUnit: unit = ""

    if subP is None: plot = plt.gca()
    else: plot = subP
    plot.yaxis.set_major_formatter(
                        ticker.FuncFormatter(
                                lambda y, pos:formatStr.format(y/dvsr)+unit
                                )
                        )

    dvsrStr = f'x10^{len(str(int(dvsr)-1))}'
    return dvsrStr 

#-------------------------------------------------------------------------------

def calcPeriods(fs, dPtIntvSec, flagXFlip):
    """ Calculate periods with frequencies returned from Welch's method PSD

    Args:
        fs (numpy.ndarray): Frequency.
        dPtIntvSec (int): Seconds used for bundling data.
        flagXFlip (bool): Flip x-axis (so that period increases)

    Returns:
        periods (list): Period list.
        pIdx2h (int): Index where period <= 2 hours.
        pIdx20m (int): Index where period < 20 minutes.
    """
    if DEBUG: logging.info(str(locals()))

    periods = []
    pIdx2h = -1 # index where period <= 2 hours 
    pIdx20m = -1 # index where period < 20 minutes 
    for fi, freq in enumerate(fs):
        period = 1.0 / freq * dPtIntvSec
        periods.append(period)
        if pIdx2h == -1:
            if flagXFlip:
                if period > (60 * 60 * 2): pIdx2h = copy(fi-1)
            else:
                if period < (60 * 60 * 2): pIdx2h = copy(fi-1)
        if pIdx20m == -1:
            if flagXFlip:
                if period > (60 * 20): pIdx20m = copy(fi-1)
            else:
                if period < (60 * 20): pIdx20m = copy(fi-1)
    return periods, pIdx2h, pIdx20m

#-------------------------------------------------------------------------------

def genXTickLbls(fs, periods, nTicks=-1, flagXFlip=False):
    """ generates x-tick labels (period at each frequency) for PSD graph.

    Args:
        fs (numpy.ndarray): Frequency.
        periods (list): Period list.
        nTicks (int): Number of ticks to display.
        flagXFlip (bool): Flip x-axis (so that period increases)

    Returns:
        x (list): List of x-positions for each label.
        xtL (list): List of x-tick labels.
    """
    if DEBUG: logging.info(str(locals()))

    def sec2txt(sec):
        if sec <= 60: txt = "%is"%(sec)
        elif 60 < sec <= 60*60: txt = "%im"%(int(np.round(sec/60)))
        else: txt = "%ih"%(int(np.round(sec/60/60)))
        return txt

    if nTicks == -1:
        if flagXFlip:
            x = [0]
            xtL = [sec2txt(periods[0])]
            sec2disp = [20*60, 2*60*60]
        else:
            x = []
            xtL = []
            sec2disp = [2*60*60, 20*60]
        for i, per in enumerate(periods):
            if len(sec2disp) == 0: break
            _flag = False
            if flagXFlip:
                if per > sec2disp[0]: _flag = True
            else:
                if per < sec2disp[0]: _flag = True
            if _flag:
                x.append(i-1)
                xtL.append(sec2txt(periods[i-1]))
                sec2disp.pop(0)
        if not flagXFlip:
            x.append(len(fs)-1)
            xtL.append(sec2txt(periods[-1])) # last label
        return x, xtL 
    
    else: 
        xtL = [] # list of x-tick labels
        step = int(round(len(periods)/nTicks))
        for nti in range(nTicks):
            period = periods[nti*step]
            xtL.append(sec2txt(period))
        x = [0] + list(range(step, len(fs), step))
        return x, xtL

#-------------------------------------------------------------------------------

def genYTickLbls(psdMV, nTicks=5):
    """ [OBSOLETE] generates y-tick labels for PSD graph.

    Args:
        psdMV (int): Max. value of y-axis in PSD graph. 
        nTicks (int): Number of ticks to display.

    Returns:
        y (list): List of y-positions for each label.
        ytL (list): List of y-tick labels.
    """
    if DEBUG: logging.info(str(locals()))

    y = []
    ytL = [] # list of y-tick labels
    step = int(psdMV/nTicks)
    dvsr = 1e3 
    for yti in range(nTicks-1):
        if yti == 0: y.append(step)
        else: y.append(y[-1]+step)
        ytL.append("%.1f"%(y[-1]/dvsr))
        #ytL.append(str(y[-1]))
    y.append(psdMV)
    ytL.append(str(psdMV))

    return y, ytL

#-------------------------------------------------------------------------------

def drawPSD(d, ax=None, title="", params={}, logFP="log.txt"):
    """ Drawing power spectral density plot

    Args:
        d (list): data with which PSD will be drawn
        ax (None/ matplotlib.axis): axis of subplot
        title (str): title of the plot
        params (dict): parameters

    Returns:
        fs (numpy.ndarray): frequency of PSD
        pxx (numpy.ndarray): power of PSD
        psdMV (float): maximum value of result power 
    """
    if DEBUG: logging.info(str(locals()))

    # default paramters 
    pa = dict(
                nXTicks = -1, #number of ticks on x-axis
                dPtIntvSec = 300, # data point interval in seconds
                maxY = -1, # max value of power in y-axis
                flagDrawPlot = True, # draw PSD graph
                flagDrawRegLn = True, # draw regression line
                flagXFlip = True, # Flip x-axis (so that period increases)
                )
    for k in params.keys(): pa[k] = params[k] # update with given parameters

    color = setColors()

    fs, pxx = signal.welch(d)
    fs = fs[1:]
    pxx = pxx[1:]
    pxx = np.sqrt(pxx) # power unit = RMS
    if pa["flagXFlip"]:
        fs = np.flip(fs)
        pxx = np.flip(pxx)
    psdMV = int(np.ceil(np.max(pxx)))

    ### compare low & high frequency range
    periods, pIdx2h, pIdx20m = calcPeriods(fs, 
                                           pa["dPtIntvSec"], 
                                           pa["flagXFlip"])
    if np.max(pxx) > 10**5: dvsr = 10**5
    else: dvsr = 1
    if pa["flagXFlip"]:
        m2hr = np.mean(pxx[pIdx20m:pIdx2h]) / dvsr
        m20m = np.mean(pxx[:pIdx20m]) / dvsr
    else:
        m2hr = np.mean(pxx[pIdx2h:pIdx20m]) / dvsr
        m20m = np.mean(pxx[pIdx20m:]) / dvsr
    msg = "mean value [2hr-20min]: %.3f"%(m2hr)
    msg += ", mean value [20min-]: %.3f"%(m20m)
    msg += ", fraction [2hr-20min]/[20min-]: %.3f"%(m2hr/m20m)
    print(msg)
    writeFile(logFP, msg)

    if pa["flagDrawPlot"]:
        msg = ""
        x, xtL = genXTickLbls(fs, periods, pa["nXTicks"], pa["flagXFlip"]) 
        ax.set_xticks(x, xtL, fontsize=8)
         
        # draw plot
        ax.plot(pxx, linewidth=1, c=color["dLn"])
        ax.set_title(title, size=8)

        ### write info about [2hr-20min] & [20min-10min]
        _xLen = len(fs)
        _x = [int(_xLen*0.25), int(_xLen*0.5), int(_xLen*0.75)]
        if pa["maxY"] == -1: _y = [0, int(psdMV*0.97), 0]
        else: _y = [0, int(pa["maxY"]*0.97), 0]
        if pa["flagXFlip"]:
            _txt = ["%.i"%(int(np.round(m20m))), 
                    "x%.6f"%(m2hr/m20m),
                    "%.i"%(int(np.round(m2hr)))]
        else:
            _txt = ["%.i"%(int(np.round(m2hr))), 
                    "x%.6f"%(m2hr/m20m),
                    "%.i"%(int(np.round(m20m)))]
        for _i in range(len(_x)):
            ax.text(_x[_i], _y[_i], _txt[_i], fontsize=8,
                    horizontalalignment="center", verticalalignment="top")

        if pa["flagDrawRegLn"]:
            ### regression line of 2hr-20min and 20min-10min
            slopes = []
            if pa["flagXFlip"]:
                #idxLst = [(pIdx20m, pIdx2h), (0, pIdx20m)]
                idxLst = [(0, pIdx2h)]
            else:
                #idxLst = [(pIdx2h, pIdx20m), (pIdx20m, len(pxx))]
                idxLst = [(pIdx2h, len(pxx))]
            for _i, rng in enumerate(idxLst):
                _x = np.asarray(list(range(rng[0], rng[1])))
                slope, intcpt = np.polyfit(_x, pxx[rng[0]:rng[1]], 1)
                ax.plot(_x, _x*slope+intcpt, c=color["rLn%i"%(_i)], 
                        linewidth=0.75)
                slopes.append(slope)
            #msg += "\nSlopes [20min-2hr] & [20min-10min]: %s"%(str(slopes))
            #msg += "\nSlope ratio [2hr-20min]:[20min-10min] is"
            #msg += " %.3f"%(slopes[0]/slopes[1])
            msg += "\nSlope of 10min-2hr: %s"%(str(slopes).strip("[]"))
 
        print(msg)
        writeFile(logFP, msg)

    ret = dict(fs=fs, pxx=pxx, psdMV=psdMV, m2hr=m2hr, m20m=m20m,
               pIdx2h=pIdx2h, pIdx20m=pIdx20m)
    return ret 

#-------------------------------------------------------------------------------

def drawDurGraphs(nw, acdS, indS, flagSetYTicks=True):
    """ Draw activity & inactivity duration graphs. 
    - Used for n1610 study (Jinook 2022)

    Args:
        nw (int): n of workers in an aggregate.
        acdS (list): List of activity durations from simulated data
        indS (list): List of inactivity durations from simulated data
        flagSetYTicks (bool): Whether to manually set y-ticks

    Returns:
        rGraph (dict): Graph images
    """
    if DEBUG: logging.info(str(locals()))

    rGraph = {}
    
    dpi = 200 
    plt.rcParams["figure.dpi"] = dpi 
    plt.rcParams["savefig.dpi"] = dpi 
    plt.rcParams["figure.figsize"] = (4, 2)
    plt.rcParams["font.size"] = 8 
    plt.rcParams["lines.linewidth"] = 1 
    
    color = setColors()

    for mk in ["activitysec", "inactivitysec"]: 
    
        nRows = 1
        nCols = 1 
        fig, axs = plt.subplots(nRows, nCols, sharey=True, sharex=True)
        if mk == "activitysec": d = acdS
        elif mk == "inactivitysec": d = indS
        ### draw box-plot
        bpLW = 0.5
        boxprops = dict(linewidth=bpLW)
        flierprops = dict(marker=",", linewidth=bpLW, markersize=1,
                          markerfacecolor=color["flier"])
        whiskerprops = dict(linewidth=bpLW)
        capprops = dict(linewidth=bpLW)
        medianprops = dict(linestyle="solid", linewidth=bpLW, 
                           color=color["medianLn"])
        plt.boxplot(d, boxprops=boxprops, flierprops=flierprops, 
                   whiskerprops=whiskerprops, capprops=capprops, 
                   medianprops=medianprops)
        plt.tick_params(labelbottom = False, bottom = False)
 
        if mk == "activitysec": dvsr = 1000
        elif mk == "inactivitysec": dvsr = 100
        title = mk.rstrip("sec").capitalize()
        title += " durations [n%i]"%(nw)
       
        maxY = int(max([max(x) for x in d]))
        if flagSetYTicks:
            ### set y-axis labels
            step = int(maxY / 4)
            maxY += int(step/2)
            plt.ylim(-int(step/2), maxY)
            y=[0]; ytL=["0"]
            for _y in range(step, maxY, step):
                y.append(_y)
                ytL.append(int(np.round(_y/dvsr)))
            plt.yticks(y, ytL, fontsize=6)

            _txt = "dur. (x1e%i sec.)"%(len(str(dvsr))-1)
        else:
            _txt = "dur."

        dvsrStr = setYFormat(maxY, None, "{:,.1f}", False)
        _txt += f' ({dvsrStr})'

        plt.gcf().text(0.01, 0.92, _txt, fontsize=8)
        plt.gcf().text(0.5, 0.03, "colonies", fontsize=8)

        plt.suptitle(title, fontsize=8)
        
        rGraph[mk] = convt_mplFig2npArr(fig)

    return rGraph

#-------------------------------------------------------------------------------

def proc_mComp(wxFrame, data, keyLst, dPtIntvSec, flagN1FromN6=True):
    """ Compare motion with given data
    - Used for n1610 study (Jinook 2022)

    Args:
        wxFrame (wx.Frame): Caller frame.
        data (list): Each item is a list of intensity data read from a data file
        keyLst(list): List of string key for each item in 'data'
        dPtIntvSec (int): Interval in seconds of data bundling
        flagN1FromN6 (bool): If True, calculate n1 data from averaging n6 data

    Returns:
        rGraph (dict): Graph images
    """
    if DEBUG: logging.info(str(locals()))
     
    ### output data labels
    if flagN1FromN6: nLblLst = ["n06", "n10"] 
    else: nLblLst = ["n01", "n06", "n10"]

    ##### [begin] prep. data -----
    sumNM = {} # sum of number of motions
    sumNM_L = {} # linear increase with n1 * number of workers
    meanPI = {} # mean power values in period range of interest
    meanPI_L = {} # linear increase with n1 * number of workers
    for di, d in enumerate(data):
        dtK, nLbl = keyLst[di].split("_") # datetime and n-label (n1, n6, n10)
        nLbl = nLbl.lower()
        if len(nLbl) == 2: nLbl = "n0" + nLbl[-1]
        nw = int(nLbl.replace("n","")) # n of workers
        if not dtK in sumNM.keys():
            sumNM[dtK] = dict(n01=None, n06=None, n10=None) 
            sumNM_L[dtK] = dict(n01=None, n06=None, n10=None) 
            meanPI[dtK] = dict(n01=None, n06=None, n10=None) 
            meanPI_L[dtK] = dict(n01=None, n06=None, n10=None) 
        # store sum of number of motions
        sumNM[dtK][nLbl] = np.sum(d)

        ### [2022.12.13] 
        ### make n01 data with n06 data, instead of using actual n01 data
        ###   (to avoid individual difference of n01)
        if flagN1FromN6 and nLbl == "n06":
            sumNM[dtK]["n01"] = int(np.round(sumNM[dtK]["n06"]/6))

        # store linear increase of n1
        sumNM_L[dtK][nLbl] = sumNM[dtK]["n01"] * nw

        params = dict(dPtIntvSec=dPtIntvSec, flagDrawPlot=False)
        ret = drawPSD(d, None, "", params)
        # store mean power values 
        #   where the period in the rage of 2 hour - 20 minutes.
        meanPI[dtK][nLbl] = ret["m2hr"]

        ### [2022.12.13] 
        if flagN1FromN6 and nLbl == "n06":
            meanPI[dtK]["n01"] = int(np.round(meanPI[dtK]["n06"]/6))

        # store linear increase of n1
        meanPI_L[dtK][nLbl] = meanPI[dtK]["n01"] * nw
    ##### [end] prep. data -----

    ##### [begin] drawing graph -----
    ### set number of columns and rows
    if wxFrame is not None:
        pa = wxFrame.mPa
        nCol = pa["gNCol"] 
        nRow = pa["gNRow"] 
    else:
        nDTK = len(sumNM)
        nCol = 4
        if nDTK <= nCol:
            nCol = 2
            nRow = 2
        else:
            nRow = int(np.ceil(nDTK/nCol)) 

    xtP = [0] + list(range(1, len(nLblLst))) # x-tick positions
    colors = setColors() 
   
    rGraph = {}
    #for dataK in ["NM", "MP"]: 
    for dataK in ["NM"]: # draw graph only Number of Motions [2022.11.20]
    # number of motions & mean power
        fig, ax = plt.subplots(nRow, nCol, sharey=True, sharex=True, 
                               tight_layout=True) 
        ri = 0
        ci = 0
        minYLst = []
        maxYLst = []
        for dtK in sorted(sumNM.keys()):
            if dtK[-1] == "s": # this is simulated data
                _title = "simulation %i"%(int(dtK[:-1]))
            else: # from real data
                _title = "%s-%s-%s"%(dtK[:4], dtK[4:6], dtK[6:8]) 
            ax[ri,ci].set_title(_title, fontsize=8)

            # data for graph; 
            #   NM=Number of motions, MP=Mean power
            #   L=linear increase of N1, M=measured
            d4g = dict(NM=dict(L=None, M=None), MP=dict(L=None, M=None))
            
            ### draw data lines of L (linear increase) & M (measured)
            for lOrM in ["L", "M"]:
            # linear increase of n1 & measured
                if lOrM == "L":
                    if dataK == "NM":
                        _d = sumNM_L
                        _lbl = "number of motions [Linear inc. of n01]"
                    elif dataK == "MP":
                        _d = meanPI_L
                        _lbl = "mean power in 2hr-20m [Linear inc. of n01]"
                    _lSty = "dotted"
                elif lOrM == "M":
                    if dataK == "NM":
                        _d = sumNM
                        _lbl = "number of motions [Measured]"
                    elif dataK == "MP":
                        _d = meanPI
                        _lbl = "mean power in 2hr-20m [Measured]"
                    _lSty = "solid"
                d4g[dataK][lOrM] = [_d[dtK][nLbl] for nLbl in nLblLst]
                ax[ri,ci].plot(d4g[dataK][lOrM], label=_lbl, linestyle=_lSty, 
                               c=colors[dataK], linewidth=1) # draw plot 
                minYLst.append(np.min(d4g[dataK][lOrM]))
                maxYLst.append(np.max(d4g[dataK][lOrM]))
            #ax[ri,ci].legend(loc="upper left") # draw legend
            ax[ri,ci].set_xticks(xtP, nLblLst, fontsize=8) # set x-ticks

            ### write text; ratio of (measured : linearIncreaseOfN1)
            for i in range(len(nLblLst)):
                _d = d4g[dataK]
                _x = i/(len(nLblLst)-1)
                _y = 0.0
                _txt = "%.6f"%(_d["M"][i]/_d["L"][i])
                if i == 0: _hA = "left"
                elif i == len(nLblLst)-1: _hA = "right"
                else: _hA = "center"
                ax[ri,ci].text(_x, _y, _txt, color=colors[dataK], fontsize=6,
                               horizontalalignment=_hA,
                               verticalalignment="bottom",
                               transform=ax[ri,ci].transAxes) 

            ci += 1
            if ci >= nCol: ci = 0; ri += 1
      
        #yRng = (int(min(minYLst)), int(max(maxYLst)))
        #plt.ylim(yRng[0], yRng[1]) # set y-axis range
        maxY = int(max(maxYLst))
        setYFormat(maxY) 

        if dataK == "NM": plt.suptitle("Number of motions")
        elif dataK == "MP": plt.suptitle("Mean power in 20-120 m.")
        rGraph[dataK] = convt_mplFig2npArr(fig)
    ##### [end] drawing graph -----
    return rGraph 

#-------------------------------------------------------------------------------

def drawProbDistributions(kwa):
    """ draw graphs of probability distributions 

    Args:
        kwa (dict):
            mpl (list): List of number of motion points
            mpp (list): List of probabilities of # of motion points
            acdp (numpy.ndarray): # probabilities of occurrences of 
                                  #   each activity duration
            acdpXLim (int): x value limit; -1 will use length of acdp
            iadp (numpy.ndarray): # probabilities of occurrences of 
                                  #   each inactivity duration
            iadpXLim (int): x value limit; -1 will use length of iadp 
            drp (numpy.ndarray): # probabilities of occurrences of 
                                 #   each direction (1-360) in movements
            dsp (numpy.ndarray): # probabilities of occurrences of 
                                 #   each distance (body-length) in movements
            dist2c (numpy.ndarray): # probabilities of distances between
                                    #   motion point and ROI0-center 
    
    Returns:
        rGraph (dist): resultant graphs 
    """
    if DEBUG: MyLogger.info(str(locals()))

    rGraph = {}

    for k in kwa.keys():
        if k == "mpl" or "xlim" in k.lower() or "ylim" in k.lower(): continue
        
        if k == "mpp": 
        # n of motion points graph 
            if "mpl" not in kwa.keys(): continue
            x = kwa["mpl"]
            _title = "Probability for number of motions"
            rK = "mp"

        else:
            if k == "acdp":
            # activity duration probability graph
                _title = "Activity duration probability"
                rK = "adp"
            
            elif k == "iadp":
            # inactivity duration probability graph
                _title = "Inactivity duration probability"
                rK = "idp"

            elif k == "drp":
                _title = "Direction probability"
                rK = "drp"

            elif k == "dsp":
                _title = "Distance probability"
                rK = "dsp"
            
            elif k == "dist2c":
                _title = "Distance between motion-point and ROI-center"
                rK = "dist2c" 

            else:
                continue

            if f'{k}XLim' in kwa.keys(): x = list(range(kwa[f'{k}XLim']))
            else: x = list(range(len(kwa[k])))

        y = kwa[k]
        fig, ax =  plt.subplots()
        if f'{k}YLim' in kwa.keys(): ax.set_ylim(0, kwa[f'{k}YLim'])
        ax.bar(x, height=y[:len(x)])
        ax.set_title(_title, size=8)
        
        if f'vline' in kwa.keys(): # if there're vertical lines to draw
            # draw the vertical lines
            for _x, _c in kwa["vline"]: ax.axvline(x=_x, color=_c)
        
        if f'xtraD' in kwa.keys(): # if there're extra data to draw
            ax.plot(kwa["xtraD"][0], color=kwa["xtraD"][1])

        rGraph[rK] = convt_mplFig2npArr(fig)

    return rGraph

#-------------------------------------------------------------------------------

if __name__ == "__main__": pass

