import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import tarfile
import os
from torchvision.io import read_image
import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from GSaddons import *

st.title('Golf Swing')
def Findbodypos( points1):
    import numpy as np
    
#     Given points returns variables for body pos- mainly to seperate left from right
    
    
    #       0      1      2     3       4      5           6         7       8       9      10     11     12    13     14
        
    body=['head','head','head','head','head','shoulder',
              'shoulder','elbow','elbow','hand','hand','hip','hip','knee','knee',
    # 15     16
    'heel','heel']

    bodyPos=[]
    # position of head tae point closest top left corner-> min x and min y
    Bpos=np.array([ ii for ii,kk in enumerate(body) if kk=='head'])
    bodyPos.append(Bpos[np.sum(points1[Bpos][:,:],axis=1)==np.min(np.sum(points1[Bpos][:,:],axis=1))][0])
    
    # scroll through body parts not the head
    bodyPart = np.unique(body)
    ii=1
    bodyPartUse=['head']
    for bb in bodyPart:
        if bb!='head':
            Bpos=np.array([ ii for ii,kk in enumerate(body) if kk==bb])

            temp=points1[Bpos,0]
          
            bodyPos.append( Bpos[temp==np.min(temp)][0] )
            bodyPartUse.append(bb+'L')
            bodyPos.append( Bpos[temp==np.max(temp)][0] )
            bodyPartUse.append(bb+'R')
    
    return bodyPartUse,bodyPos
       
def plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,marks ):
    import matplotlib.pyplot as plt
    import numpy as np
    
#     Given what to plot e.g. whattoPlot=['heelL','kneeL','hipL','shoulderL','head']
#     options=['head','shoulderL','shoulderR','elbowL','elbowR','handL','handR','hipL','hipR','kneeL','kneeR',
#     # 15     16
#     'heelL','heelR']
#     marks e.g. 'kx-'
#     points1 from model
#     Other variables from Findbodypos
    
    vec=[]
    for xx in whattoPlot:
        vec.append([bodyPos[ii] for ii,xa in enumerate(bodyPartUse) if xa ==xx][0])
    plt.plot(points1[vec,0],points1[vec,1], marks,markersize=10,linewidth=3)
    

def angleUse(pos1, pos2):
    import numpy as np
    o=(pos1[1]-pos2[1])#up
    a=(pos1[0]-pos2[0])
    try:
        ang=int((180/np.pi)*np.arctan(o/a))
    except:
        ang=0
    return ang

def doPlotMulti(imgTensName,LEG,pointsAll):
    # This to plot points
    
    plt.gca().invert_yaxis()

    
    
def doPlot(imgSEL,imgTensName,LEG,pointsAll):

    numSEL=[oo for oo,x in enumerate(imgTensName) if x==imgSEL][0]
    

    points1=pointsAll[numSEL]

    fig=plt.figure(figsize=(7,7))
    
    img = mpimg.imread(uploaded_files[numSEL])
    plt.imshow(img)
        
   
    plt.legend(LEG)
    
    for x in points1:
        plt.plot(x[0],x[1],'+b')
    
    bodyPartUse,bodyPos=Findbodypos( points1)
    whattoPlot=['heelL','kneeL','hipL','shoulderL','head']
    plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,'ko-' )

    whattoPlot=['heelR','kneeR','hipR','shoulderR','head']
    plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,'ks-' )

    whattoPlot=['handL','elbowL','shoulderL']
    plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,'mo--' )

    whattoPlot=['handR','elbowR','shoulderR']
    plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,'mo--' )
    
    whattoPlot=['hipL','hipR']
    plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,'ro--' )

    whattoPlot=['shoulderL','shoulderR']
    plotBodyPart(whattoPlot,bodyPos, bodyPartUse, points1,'ro--' )


    st.pyplot(fig)
    
   

    
@st.cache(allow_output_mutation=True)
def load_data():
    imgTens=[]
    imgTensName=[]
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()

        number_img = Image.open(uploaded_file)
        number_img = number_img.transpose(Image.ROTATE_90)

        convert_tensor = transforms.ToTensor()
        number_img=convert_tensor(number_img)
        imgTens.append(number_img)

        imgTensName.append(uploaded_file.name)

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    #     try:
    predictions=model(imgTens)
    
    vects = np.array([[ 5,6],#shoulders also 4?
     [11,12], #hips
     [13,14], #knees
     [15,16],#heels
     [7,8],#elbows
     [9,10],#hands
     ])
    pointsAll=[]
    LEG=['Back','Front','Shoulders','Hips','Knees','Heels','Elbows','Hands']
    vectsNames=LEG[2:]
#     df=pd.DataFrame({'Body':vectsNames})

    for numSEL,imgs in enumerate(imgTensName):
        points1=np.array([x.detach().numpy()[0:2] for x in predictions[numSEL]['keypoints'][0]])

        pointsAll.append(points1)

        for iv,v in enumerate(vects):
            posA=points1[v[0],:]
            posB=points1[v[1],:]

    return imgTensName, uploaded_files, pointsAll,LEG

uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True)

if len(uploaded_files) !=0:
    imgTensName, uploaded_files, pointsAll,LEG=load_data()

    
    # An optionbox- Select How search
    imgSEL = st.sidebar.selectbox(
        'Select how to search',
         imgTensName)

    
    doPlot(imgSEL,imgTensName,LEG,pointsAll)
    

#     st.sidebar.dataframe(df)
    

    
