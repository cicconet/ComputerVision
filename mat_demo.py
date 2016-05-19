# mat.py demo

import numpy as np
import cv2
from mat import *

def normalize(I):
	I = I-np.min(I)
	I = I/np.max(I)
	return I

stretch = 0
scale = 1
orientation = 180
npeaks = 1

I = normalize(cv2.imread('C1.png',0).astype('float32'))

Gmag,Gdir,BGR = mat(I)

nr,nc = np.shape(I)
L = legend(nr,nc)

S = 0.5*np.ones([nr,5,3]);
J = np.concatenate((BGR,S,L),axis=1)
cv2.imshow('Magnitudes and Gradients',J)
cv2.waitKey(0)
cv2.destroyAllWindows()