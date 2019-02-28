#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


sift1 = cv2.xfeatures2d.SIFT_create(nOctaveLayers=8)
sift2 =cv2.xfeatures2d.SIFT_create(nOctaveLayers=8)


# In[3]:


img1 = cv2.imread('../dataset/arch/01.png')
print(img1.shape)
# img1 = cv2.resize(img1,(,120))
img2 = cv2.imread('../dataset/arch/02.png')
# img2 = cv2.resize(img2,(120,120))
print(img2.shape)


# In[4]:


# kp, des = sift.detectAndCompute(gray,None)


# In[5]:


step_size = 5
kp1 = [cv2.KeyPoint(x, y, step_size) for y in range(0, img1.shape[0], step_size) 
                                    for x in range(0, img1.shape[1], step_size)]
kp2 = [cv2.KeyPoint(x, y, step_size) for y in range(0, img2.shape[0], step_size) 
                                    for x in range(0, img2.shape[1], step_size)]


# In[10]:


dense_feat1 = sift1.compute(img1, kp1)
dense_feat2 = sift2.compute(img2, kp2)


# In[11]:


print(len(kp1))
print(len(kp2))


# In[12]:


print(dense_feat1[1].shape)
print(dense_feat2[1].shape)
dense_feat1 = dense_feat1[1]
dense_feat2 =dense_feat2[1]


# In[13]:


# f = np.column_stack((dense_feat1[1],dense_feat2[1]))
# print(f.shape)


# In[ ]:


sigma_f = 1
w1 = np.array([[np.exp(-np.linalg.norm(i-j)**2/sigma_f**2) for j in dense_feat1] for i in dense_feat1])
w2 = np.array([[np.exp(-np.linalg.norm(i-j)**2/sigma_f**2) for j in dense_feat2] for i in dense_feat2])
c = np.array([[np.exp(-np.linalg.norm(i-j)**2/sigma_f**2) for j in dense_feat2] for i in dense_feat1])


# In[ ]:


t1 = np.column_stack((w1,c))
t2 = np.column_stack((np.transpose(c),w2))
W = np.row_stack((t1,t2))


# In[ ]:


u,s,v = np.linalg.svd(W)


# In[ ]:


k=5
smallest_ev = v[:,len(v)-k-1:len(v)-1]
i = np.column_stack((w1,w2))
t = np.matmul(i,smallest_ev[:,-1])
t = np.reshape(t,(img1.shape[0],img1.shape[1]))
# t[:,:,1] = np.matmul(i[:,:,1],smallest_ev[:,-1])
# t[:,:,2] = np.matmul(i[:,:,2],smallest_ev[:,-1])

cv2.imshow('k',t)
if(cv2.waitKey(0)&0xff==ord('q')):
    cv2.destroyAllWindows()

