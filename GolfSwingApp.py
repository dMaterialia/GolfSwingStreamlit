import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import tarfile
import os
from torchvision.io import read_image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


st.title('Golf Swing')

@st.cache()
def load_data(choi):

    cda = os.getcwd()
    cda2=cda
    my_tar = tarfile.open(cda2+'/GC2.tgz')
    my_tar.extractall(cda2) # specify which folder to extract to
    my_tar.close()

    imgAll=[]
    vidAll=[]
    i=0
    last1=' '
    for xx in os.listdir(cda2+'/images/'):
        if xx[-1]=='g':
            imgAll = np.append(imgAll, xx)
            if xx.split('_')[1]!=last1:
                i=i+1
            vidAll=np.append(vidAll,i)
            last1=xx.split('_')[1]

    vidAllUnq=np.unique(vidAll)
    

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    imgs = imgAll[vidAll==vidAllUnq[choi]]

    # make sure in correct order
    aa=[int(xx.split('_')[-1].split('e')[1].split('.')[0]) for xx in imgs]
    ind=sorted(range(len(aa)), key=lambda k: aa[k])
    imgs=imgs[ind]

    # create list of image locs
    imgTens=[]
    imgLocAll=[]
    iiUse=[0,3,5]
    for ii,image_filename in enumerate(imgs):
    #             print(cda2+'images/'+image_filename)
        if ii in iiUse:
            number_img = Image.open(cda2+'/images/'+image_filename)
            convert_tensor = transforms.ToTensor()
            number_img=convert_tensor(number_img)
            imgTens.append(number_img)
            imgLocAll.append(image_filename)

    predictions=model(imgTens)
    
    return predictions,imgLocAll,cda2

data_load_state = st.text('Loading data...')
useME = 3
predictions,imgLocAll,cda2=load_data(useME)
data_load_state.text("Loaded data (using st.cache)")


choice=imgLocAll

# An optionbox- Select How search
imgSEL = st.sidebar.selectbox(
    'Select how to search',
     choice)

imgLocAll
useME

img = mpimg.imread(cda2+'/images/'+imgSEL)

numSEL=[oo for oo,x in enumerate(choice) if x==imgSEL][0]

SwingPos=['Start','Back','Through']
SwingPos[numSEL]
points1=np.array([x.detach().numpy()[0:2] for x in predictions[numSEL]['keypoints'][0]])

fig=plt.figure(figsize=(7,7))
plt.imshow(img)


adjPts=1
#back of body
v=[4,6,12,14,16]
plt.plot(points1[v,0]*adjPts,points1[v,1]*adjPts, '-w<',markersize=10,linewidth=2)

#front of body
v=[0,5,11,13,15]
plt.plot(points1[v,0]*adjPts,points1[v,1]*adjPts, '-k>',markersize=10,linewidth=2)


vects = np.array([[ 5,6],#shoulders also 4?
     [11,12], #hips
     [13,14], #knees
     [15,16],#heels
     [7,8],#elbows
     [9,10],#hands
     ]) 
mak='gcyrmb'
for iv,v in enumerate(vects):
    plt.plot(points1[v,0]*adjPts,points1[v,1]*adjPts, '-'+mak[iv],markersize=10,linewidth=3)

LEG=['Back','Front','Shoulders','Hips','Knees','Heels','Elbows','Hands']
plt.legend(LEG)
for x in points1:
    plt.plot(x[0],x[1],'+b')
    
st.pyplot(fig)
