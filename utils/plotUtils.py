import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Localizer import Localizer
import scipy
import math

def plotBoxes(centroids, SEM, annotations=None, saveImg=False, saveImgPath=None):

    box_size = [100,100]
    plt.rcParams['figure.figsize'] = (16, 10)
    _, ax = plt.subplots(1)
    fig = plt.imshow(SEM,cmap='gray')
    plt.xticks([])
    plt.yticks([])

    for i in range(len(centroids)):
        rect = patches.Rectangle((centroids[i][1]-box_size[1]/2.,centroids[i][0]-box_size[0]/2.),
                                box_size[0],box_size[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    if annotations!=None:
        for i in range(len(annotations)):
            plt.text(centroids[i][1]-box_size[1]/2, centroids[i][0]+box_size[0]/2+45,
                    annotations[i],fontsize=14,color='red')

    if saveImg:
        plt.savefig(saveImgPath,bbox_inches='tight')

    plt.show()
    
if __name__=='__main__':
    eps = 4
    min_samples = 20
    theta = 5
    imgpath = "/home/tom/Data/IN-SITU/15/PanoramaSmallX_4Y_1.png"
    
    SEM = scipy.misc.imread(imgpath,flatten=True)

    L = Localizer(eps, min_samples, theta)
    centroids = L.predict(SEM)
    
    size = SEM.shape
    
    window_size = [250,250]
    
    for i in range(len(centroids)):
        x1 = math.floor(centroids[i][0] - window_size[0]/2)
        y1 = math.floor(centroids[i][1] - window_size[1]/2)
        x2 = math.floor(centroids[i][0] + window_size[0]/2)
        y2 = math.floor(centroids[i][1] + window_size[1]/2)
        
        if y2-y1==249:
            print y1, y2
            print centroids[i][1]
            print window_size[1]/2
        
        ##
        ## Catch the cases in which the extract would go
        ## over the boundaries of the original image
        ##

        if x1<0:
            x1 = 0
            x2 = window_size[0]
        if x2>=size[0]:
            x1 = size[0] - window_size[0]
            x2 = size[0]
        if y1<0:
            y1 = 0
            y2 = window_size[1]
        if y2>=size[1]:
            y1 = size[1] - window_size[1]
            y2 = size[1]
        
    
    plotBoxes(centroids,SEM)
