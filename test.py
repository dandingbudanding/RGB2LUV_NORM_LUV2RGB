# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:03:36 2019

@author: wfy
"""

import cv2
import numpy as np

def LRegularization(L):
    Lmax,Lmin=np.max(L),np.min(L)
    print(Lmax,Lmin)
    row,col=L.shape
    for i in range(row):
        for j in range(col):
            L[i,j]=100.0*(L[i,j]-Lmin)/(Lmax-Lmin)
    return L

def RGB2Luv(RGBPixel):
    T=np.array([[0.412453,0.357580 ,0.180423 ],
                [0.212671 ,0.715160 ,0.072169 ],
                [0.019334 ,0.119193 ,0.950227 ]])
    R=RGBPixel[2]/255.0
    G=RGBPixel[1]/255.0
    B=RGBPixel[0]/255.0
    X,Y,Z=0.0,0.0,0.0
    X,Y,Z=np.matmul(T,np.array([R,G,B]).T)

    L,u,v=0.0,0.0,0.0
    if Y>0.008856:
        L = 116.0 * np.power(Y, 1.0/3.0)-16.0
    else:
        L = 903.3 * Y
    sum= X + 15 * Y + 3 * Z
    if sum != 0:
        u = 4.0 * X / sum
        v = 9.0 * Y / sum
    else:
        u = 4.0
        v = 9.0 / 15.0


    L = L;
    u = 13 * L * (u - 0.19784977571475);
    v = 13 * L * (v - 0.46834507665248);
    return L,u,v

def Luv2RGB(LuvPixel):
    [L,u,v]=LuvPixel
    if L < 0.1:
        return 0,0,0;

    else:
        x, y, z=0.0,0.0,0.0
        if L<= 7.9996:
            y = L / 903.3

        else:
            y = (L+ 16.0) / 116.0
            y  = y * y * y
        u = u / (13 * L) + 0.19784977571475
        v = v / (13 * L) + 0.46834507665248

        x = 9 * u * y / (4 * v);
        z = (12 - 3 * u - 20 * v) * y / (4 * v);

        R = 3.240479 * x - 1.537150 * y - 0.498535 * z
        G = -0.969256 * x + 1.875992 * y + 0.041556 * z
        B = 0.055648 * x - 0.204043 * y + 1.057311 * z

        # print(B,G,R)
        R= R*255
        G=G*255
        B=B*255

        R = 0 if(R < 0) else (255 if(R>255) else R)
        G = 0 if(G < 0) else (255 if(G>255) else G)
        B = 0 if(B < 0) else (255 if(B>255) else B)

        return B,G,R

# img_src=cv2.imread('3.jpg',-1)
# img_src=img_src.astype(np.float32)
# row,col,dep=img_src.shape
# Luv=np.zeros((row,col,dep), dtype=np.float32)
# Luv=cv2.cvtColor(img_src,cv2.COLOR_BGR2Luv)
# L,u,v=cv2.split(Luv)
# L=LRegularization(L)
# Luv=cv2.merge([L,u,v])
# imgdst=cv2.cvtColor(Luv,cv2.COLOR_Luv2BGR)
# imgdst=imgdst.astype(np.int8)
# cv2.imshow('1',imgdst)
# cv2.waitKey(0)



if __name__ == "__main__":
    img_src=cv2.imread('2.jpg',-1)
    img_src=img_src.astype(np.float32)
    row,col,dep=img_src.shape
    assert(dep==3)
    Luv=np.zeros((row,col,dep), dtype=np.float32)
    img_dst=np.zeros((row,col,dep), dtype=np.float32)

    for i in range(row):
        for j in range(col):
            Luv[i,j,:]=RGB2Luv(img_src[i,j,:])

    Luv[:,:,0]=LRegularization(Luv[:,:,0])

    for i in range(row):
        for j in range(col):
            img_dst[i,j,:]=Luv2RGB(Luv[i,j,:])
    img_dst=img_dst.astype(np.uint8)
    cv2.imshow('img_dst',img_dst)
    cv2.waitKey(0)



        