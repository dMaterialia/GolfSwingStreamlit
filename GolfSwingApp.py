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

    # file names
    fls=os.listdir(cda2+'/images/')
    # frame of video
    frame=[]
    # video name
    vidAll=[]
    # file name image only 
    flsImg=[]

    for xx in fls:
        # file names that are images
        if xx[-1]=='g':
            flsImg.append(xx)
            frame.append(int(xx.split('_')[-1].split('e')[1].split('.')[0]))
            vidAll.append( xx.split('_')[1] )

    # the unique videos
    vidAllUnq=np.unique(vidAll)

    # the image name & frame for file of video used
    flsUse, frameUse=[], []
    for ii,xx in enumerate(vidAll):
        if xx==vidAllUnq[choi]:
            flsUse=np.append(flsUse, flsImg[ii] )
            frameUse=np.append(frameUse, frame[ii] )

    # make sure order is correct based on number of frame
    ind=sorted(range(len(frameUse)), key=lambda k: frameUse[k])
    flsUse=flsUse[ind]
    print(flsUse)
    # use only the 0th, 3rd and 5th value
    iiUse=[0,3,5]
    flsUse=[x for ii,x in enumerate(flsUse) if ii in iiUse]
    
    # the tensor of images
    imgTens=[]
    for ii,image_filename in enumerate(flsUse):
        number_img = Image.open(cda2+'/images/'+image_filename)
        convert_tensor = transforms.ToTensor()
        number_img=convert_tensor(number_img)
        imgTens.append(number_img)

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    predictions=model(imgTens)
    
    return predictions,flsUse,cda2

data_load_state = st.text('Loading data...')
predictions,imgLocAll,cda2=load_data(2)
data_load_state.text("Loaded data (using st.cache)")


choice=imgLocAll

# An optionbox- Select How search
imgSEL = st.sidebar.selectbox(
    'Select how to search',
     choice)


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
