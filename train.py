from models.classifier import combined_classifier
from utils.Localizer import Localizer
import scipy
import argparse
from utils.plotUtils import plotBoxes

parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=4, help="epsilon value for the DBScanner",type=int)
parser.add_argument("--min_samples", default=20, help="min_samples for the DBScanner",type=int)
parser.add_argument("--theta", default=20, help="Gray scale cut off value",type=int)
args = parser.parse_args()

eps = args.eps
min_samples = args.min_samples
theta = args.theta

imgpath = "/home/tom/FullModel/PanoramaSmall1X_1Y_3.png"



# SEM = scipy.misc.imread(imgpath,flatten=True)
# L = Localizer(eps, min_samples, theta)
# centroids = L.predict(SEM)

settingsjson = "/home/tom/FullModel/settings.json"
cc = combined_classifier(settingsjson)

# annotations = cc.predict(SEM,centroids)
# 
# plotBoxes(centroids, SEM, annotations=annotations, saveImg = True, saveImgPath=imgpath[:-4]+"_annotated.png")
