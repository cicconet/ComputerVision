# magnitudes and tangents

import numpy as np
import cv2

def mat(I,viewOutput = True):
    stretch = 0
    scale = 1
    npeaks = 1
    mi = imorlet(stretch,scale,0,npeaks)
    Gx = cv2.filter2D(I,-1,mi)
    mi = imorlet(stretch,scale,90,npeaks)
    Gy = cv2.filter2D(I,-1,mi)

    Gmag = np.sqrt(Gx*Gx+Gy*Gy)
    Gmag = Gmag/np.max(Gmag)
    
    Gdir = np.arctan2(Gy,Gx)/np.pi*180 # -180 to 180
    Gdir[np.less(Gdir,0)] = Gdir[np.less(Gdir,0)]+360 # 0 to 360

    H = Gdir
    S = np.ones(np.shape(H))
    V = Gmag

    if viewOutput:
	    nr,nc = np.shape(I)
	    HSV = np.zeros([nr,nc,3]).astype('float32')
	    HSV[:,:,0] = H
	    HSV[:,:,1] = S
	    HSV[:,:,2] = V

	    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

	    return Gmag, Gdir, BGR

    return Gmag, Gdir

def legend(nr,nc):
	L = np.zeros([nr,nc,3]).astype('float32')
	L[:,:,1] = 1
	r0 = nr/2
	c0 = nc/2
	l0 = min(nr,nc)/6
	l1 = 1.25*l0
	for a in np.arange(0,2*np.pi,np.pi/16):
	    for l in np.arange(l0,l1):
	        r = np.round(r0+l*np.cos(a))
	        c = np.round(c0+l*np.sin(a))
	        if l == l0:
	            L[r-1:r+2,c-1:c+2,0] = a/(2*np.pi)*360
	            L[r-1:r+2,c-1:c+2,2] = 1
	        else:
	            L[r,c,0] = a/(2*np.pi)*360
	            L[r,c,2] = 1
	return cv2.cvtColor(L, cv2.COLOR_HSV2BGR)

def imorlet(stretch, scale, orientation, npeaks):
    # imaginary part of morlet wavelet

    # controls width of gaussian window (default: scale)
    sigma = scale

    # orientation (in radians)
    theta = -(orientation-90)/360*2*np.pi

    # controls elongation in direction perpendicular to wave
    gamma = 1/(1+stretch)

    # width and height of kernel
    support = 2.5*sigma/gamma

    # wavelength (default: 4*sigma)
    lbda = 1/npeaks*4*sigma

    # phase offset (in radians)
    psi = 0

    xmin = -support
    xmax = -xmin
    ymin = xmin
    ymax = xmax

    xdomain = np.arange(xmin,xmax+1)
    ydomain = np.arange(ymin,ymax+1)

    [x,y] = np.meshgrid(xdomain,ydomain)

    xprime = np.cos(theta)*x+np.sin(theta)*y
    yprime = -np.sin(theta)*x+np.cos(theta)*y

    expf = np.exp(-0.5/sigma**2*(xprime**2+gamma**2*yprime**2))

    mi = expf*np.sin(2*np.pi/lbda*xprime+psi)

    # mean = 0
    mi = mi-np.sum(mi)/mi.size

    # norm = 1
    mi = mi/np.sqrt(np.sum(mi*mi))
    
    return mi