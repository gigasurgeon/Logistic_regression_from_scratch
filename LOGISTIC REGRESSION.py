# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:08:46 2018

@author: CAPTAIN
"""

import numpy as np
import pandas as pd

data=pd.read_csv("Iris.csv")
data.dropna()
x=np.array(data.iloc[:,:-1])

y=np.array(data["Species"],dtype="U5").reshape(-1,1)
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)

theta=np.zeros((x.shape[1],1))

def cost(x,y,theta):
    h=np.matmul(x,theta)
    
