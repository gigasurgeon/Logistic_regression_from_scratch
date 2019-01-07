# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 00:31:20 2018

@author: CAPTAIN
"""

import os,sys
import numpy as np
import pandas as pd
import cv2


#loading training images and test images
a=os.listdir('train')
b=os.listdir('test')

#create a zero vector to initiate stacking numpy arrays
X=np.zeros((64*64*3,1))
Y=[]

#reading and stacking training data horizontally


for i,file in enumerate(a[:200]) :
    img=cv2.imread('train//'+str(file))
    img=cv2.resize(img,(64,64))
    img=np.array(img)
    img=img.reshape(64*64*3,-1)
    X=np.hstack((X,img))
    Y.append(file[:][:3])
    print(i)
    
#normalizing images   
X=X/255
X=X[:,1:]


def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(x,y,theta):
    
    loss= (-1/m)*(y*np.log(h)+(1-y)*np.log(1-h))
    return loss

