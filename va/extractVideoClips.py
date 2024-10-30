# coding: UTF-8

"""
For extracting video clips using FFMpeg with output data from VideoAnalysis program (va.py).

Coded and tested on Ubuntu 22.04.

Jinook Oh, Acoustics Research Institute, Wien, Austria.
last edited: 2024-10-29

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

import sys, subprocess
from os import path
from datetime import timedelta

sys.path.append("..")
import initVars
initVars.init(__file__)
from initVars import *

FLAGS = dict(
                debug = False,
                )

#---------------------------------------------------------------------------

def loadCSV(fp):
    if not path.isdir(fp):
        return False, "no such directory exists."

    csvFP = path.join(fp, "rslt.csv")
    if not path.isfile(csvFP):
        txt = "Result CSV file; rslt.csv, doesn't exist in the directory."
        return False, txt

    fh = open(csvFP, "r")
    lines = fh.readlines()
    fh.close()

    bpIdx = -1
    oData = []
    for line in lines:
        items = line.split(",")
        
        if bpIdx == -1 and items[0].strip() == 'frame-index':
            for i, item in enumerate(items):
                if item.strip() == "bPresence":
                    bpIdx = i
                    break

        try: fi = int(items[0])
        except: continue

        oData.append(eval(items[bpIdx]))

    return True, oData

#---------------------------------------------------------------------------
    
def extractVideoClips(fp, data, flagOverwrite):
    """ extract video clips from the original video file 
    
    Args:
        fp (str): Output folder
        data (list): Output data (bPresence) for each frame
        flagOverwrite (bool): whether to overwrite the existing video clips
    
    Returns:
        None
    """
    if FLAGS["debug"]: MyLogger.info(str(locals()))

    def getDur(sIdx, eIdx, fIntv):
        #sIdx = max(0, sIdx-fps)
        eIdx -= nTol
        startT = timedelta(seconds = sIdx*fIntv)
        currT = timedelta(seconds = eIdx*fIntv)
        dur = currT - startT
        return startT, dur

    # get video file path
    vfp = fp.rstrip("_output")

    ### get budgie name
    bName = fp.split(".")[0].split("_")[-1]
    try: bName = int(bName)
    except: pass
    if type(bName) == int: bName = None

    fps = 30 # FPS of the input video
    fIntv = 1.0 / fps # frame interval 
    nTol = fps * 3 # n of frames to tolerate with budgie absence 
    cntAbsence = 0 # n of frames of continous budgie absence 
    timeLst = [] # list of tuples; 
                 #   (start-time, duration) for ffmpeg command
    sIdx = -1 # start-index
    totalDur = timedelta(seconds=0) 
    for fi in range(len(data)):
        if data[fi]:
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
                    if dur > timedelta(seconds=1): # cut off too short clip
                        totalDur += dur 
                        timeLst.append((sIdx, fi, str(startT), str(dur)))
                    cntAbsence = 0
                    sIdx = -1 
    if sIdx != -1:
        startT, dur = getDur(sIdx, fi, fIntv)
        totalDur += dur 
        timeLst.append((sIdx, fi, str(startT), str(dur)))

    txt = f"{len(timeLst)} clips will be extracted.\n"
    txt += f"Total duration: {str(totalDur)}"
    print(txt)

    if len(sys.argv) > 2 and sys.argv[2] == "-i":
        return

    for i, (sIdx, eIdx, sTime, dur) in enumerate(timeLst):

        st = int(sTime.replace(":","").split(".")[0])
        st = f"{st:06d}"
        if bName is None: fn = f"clip_{i:03d}_{st}.mp4"
        else: fn = f"{bName}_{i:03d}_{st}.mp4"
        outputFP = path.join(fp, fn)
        
        cmd = "ffmpeg"
        #cmd += f" -i '{vfp}' -ss {sTime} -t {dur}"
        cmd += f" -ss {sTime} -i '{vfp}' -t {dur}"
        cmd += " -c:v libx264 -crf 17 -pix_fmt yuv420p"
        if flagOverwrite: cmd += " -y"
        else: cmd += " -n"
        cmd += f" '{outputFP}'"
        cmd = cmd.split(" ")
        print(f"[{i+1}/{len(timeLst)}]\n", cmd)

        process = subprocess.Popen(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT, 
                                   universal_newlines=True,
                                   bufsize=1)

        while True:
            outputLn = process.stdout.readline()
            if outputLn == "" and process.poll() is not None:
                break
            if outputLn:
                print(outputLn.strip() + "\r", end="")
                sys.stdout.flush()

        rc = process.poll()
        if rc != 0: print(f"Failed: {rc}")
        else: print("Command executed successfully.")

#---------------------------------------------------------------------------


fp = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "-n":
    flagOverwrite = False
else:
    flagOverwrite = True
ret, output = loadCSV(fp)
if not ret: print(output)
else: extractVideoClips(fp, output, flagOverwrite)




