import scipy
from utils.Localizer import Localizer
from utils.plotUtils import plotBoxes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=4, help="epsilon value for the DBScanner",type=int)
parser.add_argument("--min_samples", default=10, help="min_samples for the DBScanner",type=int)
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

##
## Using InceptionV3 determine if the damage is an inclusion
##

weights_path = "models/InceptionV3_IncVsAll_NoInc.hdf5"
NOC = 2
network = InceptionV3.InceptionV3(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=[250,250,1],
                pooling=None,
                classes=NOC)


plotBoxes(centroids, SEM)
