# coding: UTF-8
"""
For controlling electronics connected to GPIO pins on Raspberry Pi.

Dependency:
    RPi.GPIO

last editted: 2022-05-16
"""

from os import path
from glob import glob
from time import time, sleep

import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera


DEBUG = False

#===============================================================================

class RaspGPIO:
    """ Class for interacting with sensor/actuators on GPIO pins of Raspberry Pi
    """
    def __init__(self, parent):
        if DEBUG: print("RaspGPIO.__init__()")

        ##### [begin] setting up attributes -----
        self.parent = parent
        self.classTag = "raspGPIO"
        GPIO.setmode(GPIO.BCM)
        self.gpioPin = {}
        self.gpioPin["irLed0"] = 26
        self.gpioPin["irLed1"] = 19 
        GPIO.setup(self.gpioPin["irLed0"], GPIO.OUT)
        GPIO.setup(self.gpioPin["irLed1"], GPIO.OUT)
        self.tempSensorFile= "" 
        if parent.flags["tempMeasure"]:
            oneWireDir = "/sys/bus/w1/devices/"
            tsDir = glob(oneWireDir + "28*")[0]
            self.tempSensorFile = path.join(tsDir, "w1_slave") 
            if not path.isfile(self.tempSensorFile): self.tempSensorFile= "" 
        ##### [end] setting up attributes -----

        parent.log("Mod init.", self.classTag)

    #---------------------------------------------------------------------------
    
    def logTemp(self):
        """ Read tempeture from DS18B20 temperature sensor 
        and log.
        
        Args: None
        
        Returns: None
        """
        if DEBUG: print("RaspGPIO.logTemp()")

        if self.tempSensorFile == "":
            print("Temperature sensor not available.")

        f = open(self.tempSensorFile, "r")
        lines = f.readlines()
        f.close()
        if lines == [] or not "YES" in lines[0]: return
        equalsPos = lines[1].find("t=")
        if equalsPos != -1:
            tempStr = lines[1][equalsPos+2:]
            tempC = float(tempStr) / 1000.0
            self.parent.log("Temperature: %.1f"%(tempC), self.classTag)
  
    #---------------------------------------------------------------------------
    
    def sendSignal(self, pinStr, signal):
        """ Send signal (high or low) to a GPIO pin. 
        
        Args:
            pinStr (str): String to indicate a GPIO pin. 
            signal (bool): High (True) or Low (False); GPIO.HIGH = True = 1 
        
        Returns:
            None 
        """
        if DEBUG: print("RaspGPIO.sendSignal()")

        pinNum = self.gpioPin[pinStr]
        GPIO.output(pinNum, signal) # turn off IR LED light
        ### leave a log
        msg = "Signal-%i was sent to %s (%i)."%(signal, pinStr, pinNum)
        self.parent.log(msg, self.classTag)

    #---------------------------------------------------------------------------
    
    def close(self):
        """ close module 

        Args: None
        
        Returns: None
        """
        if DEBUG: print("RaspGPIO.close()")

        GPIO.cleanup() # clean up GPIO
        self.parent.log("Mod close.", self.classTag)

    #---------------------------------------------------------------------------

#===============================================================================

class RaspCSICam:
    """ Class for reading camera attached to Raspberry Pi 
    using camera serial interface 
    """
    def __init__(self, parent, cIdx, desiredRes=(-1, -1)):
        if DEBUG: print("RaspCSICam.__init__()")

        ##### [begin] setting up attributes -----
        self.parent = parent
        self.classTag = "raspCSICam"
        self.cIdx = cIdx # cam index
        if desiredRes == (-1, -1):
            #self.res = (2592, 1944)
            #self.res = (4056, 3040) # IMX477 full resolution
            self.res = (1920, 1440)
        else:
            self.res = desiredRes
        self.fps = 30 
        ##### [end] setting up attributes -----
 
        parent.log("Mod init.", self.classTag)

    #---------------------------------------------------------------------------

    def run(self, q2m, q2t, recFolder=""):
        """ Function for thread to retrieve image
        and store it as video or image
        
        Args:
            q2m (queue.Queue): Queue to main thread to return message.
            q2t (queue.Queue): Queue from main thread.
            recFolder (str): Folder to save recorded videos/images.
        
        Returns:
            None
        """
        if DEBUG: print("RaspCSICam.run()")

        q2tMsg = '' # queued message sent from main thread
        lastFrameProcTime = time() # last frame processing time
        fpsTimeSec = time()
        fpsLastLoggingTime = time()
        fps = [0]
        flagFPSLogging = False
        lastTime = time()

        ### init PiCamera
        self.cap= PiCamera()
        self.cap.resolution = self.res 
        self.cap.framerate = self.fps 
        stream = PiRGBArray(self.cap, size=self.res)
        sleep(2)

        ##### [begin] infinite loop of thread -----
        while True:
            """ 
            ### fps
            if time()-fpsTimeSec > 1:
                if flagFPSLogging:
                    if time()- fpsLastLoggingTime > 60: # one minute passed
                        log = "FPS during the past minute %s"%(str(fps[:60]))
                        self.parent.log(log, self.classTag)
                        fpsLastLoggingTime = time()
                else:
                    print("[%s] FPS: %i"%(self.classTag, fps[-1]))
                fps.append(0)
                fpsTimeSec = time()
                # keep fps records of the past one minute
                if len(fps) > 61: fps.pop(0)
            else:
                fps[-1] += 1
            """
            msg = "[%s] Elapsed time since last frame: "%(self.classTag)
            msg += "%.3f"%(time()-lastTime)
            print(msg)
            lastTime = time()

            ### process queue message (q2t)
            if q2t.empty() == False:
                try: q2tMsg = q2t.get(False)
                except: pass
            if q2tMsg != "":
                if q2tMsg == "quit":
                    break
                q2tMsg = ""
          
            self.cap.capture(stream, format="bgr")
            frame = stream.array.copy() # frame image
            stream.truncate(0) # clear the stream for the next frame
            
            if q2m.empty():
                # send frame via queue to main
                q2m.put(["frameImg", self.cIdx, frame], True, None)

        ##### [end] infinite loop of thread -----
        self.parent.log("Thread stops.", self.classTag)    

    #---------------------------------------------------------------------------
    
    def close(self):
        """ close module 

        Args: None
        
        Returns: None
        """
        if DEBUG: print("RaspCSICam.close()")
        
        self.cap.close()
        self.parent.log("Mod close.", self.classTag)

    #---------------------------------------------------------------------------


#===============================================================================
 
