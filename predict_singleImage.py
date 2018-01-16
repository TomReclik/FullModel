import scipy
import numpy as np
from utils.Localizer import Localizer
from utils.plotUtils import plotBoxes
from models.classifier import classifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=4, help="epsilon value for the DBScanner",type=int)
parser.add_argument("--min_samples", default=20, help="min_samples for the DBScanner",type=int)
parser.add_argument("--theta", default=20, help="Gray scale cut off value",type=int)
parser.add_argument("--imgpath", help="Path to the image")
args = parser.parse_args()

eps = args.eps
min_samples = args.min_samples
theta = args.theta
imgpath = args.imgpath

##
## Read image and flatten it
##

SEM = scipy.misc.imread(imgpath,flatten=True)

##
## Using DBScan find clusters
##

L = Localizer(eps, min_samples, theta)
centroids = L.predict(SEM)

annotations = ["Not Classified" for i in range(len(centroids))]

##
## Using InceptionV3 determine if the damage is an inclusion
##

weights_path = "/home/tom/FullModel/models/InceptionV3_IncVsAll_NoInc.hdf5"
model = "InceptionV3"
window_size = [250,250]

FirstStage = classifier(weights_path, model, window_size)
y_pred = FirstStage.predict(SEM, centroids)

not_inc = []
pos_not_inc = []
threshold = 0.7
for i in range(len(centroids)):
    if y_pred[i][0]>threshold:
        annotations[i] = "Inclusion"
    else:
        not_inc.append(centroids[i])
        pos_not_inc.append(i)

weights_path = "/home/tom/FullModel/models/EERACN_SecondStage.hdf5"
model = "EERACN"
window_size = [50,50]

SecondStage = classifier(weights_path, model, window_size)
y_pred = SecondStage.predict(SEM, not_inc)

for i in range(len(not_inc)):
    if y_pred[i][0]>threshold:
        annotations[pos_not_inc[i]]="Martensite"
    elif y_pred[i][1]>threshold:
        annotations[pos_not_inc[i]]="Interface"
    elif y_pred[i][2]>threshold:
        annotations[pos_not_inc[i]]="Notch"
    elif y_pred[i][3]>threshold:
        annotations[pos_not_inc[i]]="Boundary"


plotBoxes(centroids, SEM, annotations=annotations, saveImg = True, saveImgPath=imgpath[:-4]+"_annotated.png")
