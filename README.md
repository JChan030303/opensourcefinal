# opensourcefinal
 - This program is for detecting and distinguishing brain tumors by MRI pircture.  

 # Load Pakages
 import os

import sklearn.datasets
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics

import skimage.io
import skimage.transform
import skimage.color

import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline

from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt

- Pakages that I imported

# Load Data
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

# Since there is additional test dataset, I used all the data on trainning
X_train = X
y_train = y


# Model
y_pred = np.zeros_like(y_test)

ETC = ExtraTreesClassifier(n_estimators=271, random_state = 1000)
ETC.fit(X_train, y_train)
y_pred = ETC.predict(X_test)

print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))

- Please commit the code in order
- licence: mit licence
#contact information
- william.joo03@gmail.com


