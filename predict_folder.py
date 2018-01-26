import scipy
import numpy as np
import os
import json
from utils.Localizer import Localizer
from utils.plotUtils import plotBoxes
from models.classifier import combined_classifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder with images to be annotated")
args = parser.parse_args()

folder = args.folder

eps = 4
min_samples = 20
theta = 5

L = Localizer(eps, min_samples, theta)

settingsjson = "/home/tom/FullModel/settings.json"
cc = combined_classifier(settingsjson)

for imgpath in os.listdir(folder):
    if imgpath.endswith(".png"):
        print "Processing", imgpath
        SEM = scipy.misc.imread(os.path.join(folder,imgpath),flatten=True)
        centroids = L.predict(SEM)
        annotations = cc.predict(SEM,centroids)
    
        img_dict = {}
        img_dict["ImagePath"] = imgpath
        img_dict["Annotations"] = []
        
        for i in range(len(centroids)):
            
            c = centroids[i].tolist()
            damage_site_dict = {}
            damage_site_dict["Centroid"] = c
            damage_site_dict["Class"] = annotations[i]
            img_dict["Annotations"].append(damage_site_dict)
        
        dictpath = imgpath[:-4] + ".json"
        dictpath = os.path.join(folder,dictpath)
        with open(dictpath, 'w') as outfile:
            json.dump(img_dict, outfile)
        
            
    
    
    
    
    
    
    
    
    
    
    
    
