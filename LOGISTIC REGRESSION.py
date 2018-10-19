# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:08:46 2018

@author: CAPTAIN
"""

import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
data=pd.read_csv("Iris.csv")
data.dropna()
x=np.array(data.iloc[:,:-1])

y=np.array(data["Species"],dtype="U5").reshape(-1,1)
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)
alpha=0.001

theta=np.random.randint(-1,high=5,size=(x.shape[1],1))

def cost(x,y,theta):
    h1=np.matmul(x,theta)
    sig=1/(1+np.exp(-h1))
    c1= (1/x.shape[0])*(-y*np.log(sig)+(y-1)*(np.log(1-sig)))
    return h1,c1

def gradient_descent(x,y):
    global theta
    iter=10000
    for i in range(iter):
        h2,c2=cost(x,y,theta)
        theta=theta- (alpha/x.shape[0])*(np.matmul(x,h2-y))
        
        if i%100==0:
            print(c2)
            

gradient_descent(x,y)
