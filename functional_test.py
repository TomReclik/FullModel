from models.classifier import combined_classifier
from models.classifier import classifier
from utils.Localizer import Localizer
import scipy
import argparse

def test_combined_classifier_prediction():
    eps = 4
    min_samples = 20
    theta = 20

    imgpath = "/home/tom/FullModel/PanoramaSmall1X_1Y_3.png"

    SEM = scipy.misc.imread(imgpath,flatten=True)
    L = Localizer(eps, min_samples, theta)
    centroids = L.predict(SEM)

    ##
    ## Combined classifier
    ##

    settingsjson = "/home/tom/FullModel/settings.json"
    cc = combined_classifier(settingsjson)

    annotations_combined = cc.predict(SEM,centroids)

    #
    # Combined classifier with modules executed seperately
    #

    annotations = ["Not Classified" for i in range(len(centroids))]

    weights_path = "/home/tom/FullModel/models/InceptionV3_IncVsAll_NoInc.hdf5"
    model = "InceptionV3"
    window_size = [250,250]

    FirstStage = classifier(model,2, window_size,weights_path=weights_path)
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

    SecondStage = classifier(model,4, window_size,weights_path=weights_path)
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

    assert annotations==annotations_combined, "Classifier not correctly combined"

if __name__=="__main__":
    test_combined_classifier_prediction()
    
