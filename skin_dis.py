import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
dir= "data\\"
categories =['benign','malignant']
#load the data from the folder 
data=[]
for category in categories:
    path=os.path.join(dir,category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        try:
            skin_img=cv2.imread(imgpath,0)
            image=cv2.resize(skin_img,(50,50))
            #converting the 2d array into 1d array
            image=np.array(skin_img).flatten()
            
            data.append([image,label])
        except Exception as e:
            pass
        
        
print(len(data))
#seperating the feature and label from the data for further classification
feature=[]
label=[]
for  i in range(0,len(data)):
    feature.append(data[i][0])
    label.append(data[i][1])
#used to split the train and test dataset 
x_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.7)
#this model helps to classify 
model=SVC(C=1,kernel='poly',gamma='auto')
model.fit(x_train,y_train)

pred=model.predict(x_test)

accuracy=model.score(x_test,y_test)
categories =['benign','malignant']
print('Accuracy:',accuracy)
print('prediction:',categories[pred[0]])
skin_d=x_test[0]
skin_d=cv2.resize(skin_d,(50,50))
plt.imshow(skin_d)
plt.show() 