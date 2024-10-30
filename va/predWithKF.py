# coding: UTF-8

"""
For predicting the current values, position (x, y) or size (w, h),
of an animal based on the previous datat, using Kalman filter.

Coded and tested on Ubuntu 22.04.

Jinook Oh, Acoustics Research Institute, Wien, Austria.
last edited: 2024-10-08

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

import numpy as np
from filterpy.kalman import KalmanFilter

from initVars import *
from modFFC import * 

FLAGS = dict(
                debug = False,
                )

#===============================================================================

class PredWithKalmanFilter:
    """ Class for predicting the currrent position/size of an animal subject,
    based on previous data, using Kalman-filter. 
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, params):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ##### [begin] setting up attributes -----
        defVal = dict(
                        mErr=10, 
                        mUncert=0.01, 
                        initSpat=100, 
                        initDynamic=200
                        )
        for key in defVal:
            if key not in params.keys(): params[key] = defVal[key]
        p = params
        
        # 4D state (x, y, vx, vy) and 2D measurement (x, y)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (F)
        dt = 1  # time step (1 frame)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # measurement matrix (H)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # measurement uncertainty (R)
        #   [[5, 0], [0,5]]: Expect +- 5 pixels of measurement error
        self.kf.R = np.array([
                                [p["mErr"], 0],
                                [0, p["mErr"]]
                                ])

        # process uncertainty (Q)
        #   uncertainty associated with the model's predictions and 
        #   how much change you expect between states
        self.kf.Q = np.eye(4) * p["mUncert"]

        # initial state covariance (P)
        #   uncertainty of the current state estimates based on 
        #   previous measurements
        self.kf.P = np.eye(4)
        # init.uncertainty in x (or width)
        self.kf.P[0, 0] = p["initSpat"]
        # init.uncertainty in y (or height)
        self.kf.P[1, 1] = p["initSpat"] 
        # init.uncertainty in x (or width); dynamic variable
        self.kf.P[2, 2] = p["initDynamic"] 
        # init.uncertainty in y (or height); dynamic variable
        self.kf.P[3, 3] = p["initDynamic"]
        ##### [end] setting up attributes ----- 

    #---------------------------------------------------------------------------
    
    def initState(self, d1, d2):
        """ initialize the state

        Args:
            d1 (tuple): x & y (or w & h)
            d2 (tuple): x & y (or w & h) 

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        vx = d2[0] - d1[0]
        vy = d2[1] - d1[1]
        self.kf.x = np.array([d1[0], d1[1], vx, vy])
        self.kf.predict()

    #---------------------------------------------------------------------------
    
    def update(self, d):
        """ update position (or size) data 

        Args:
            d (tuple): x & y (or w & h) of an animal in a frame

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        self.kf.update(d) # correct with measured data 
        self.kf.predict() # predict next state

    #---------------------------------------------------------------------------
    
    def getPred(self):
        """ returns the predicted data 

        Args:
            None

        Returns:
            (tuple): predicted x & y (or w & h) 
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        return (self.kf.x[0], self.kf.x[1])
    
    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass





