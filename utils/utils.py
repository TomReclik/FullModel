import os
import xml.etree.ElementTree as ET
import scipy.misc as misc
import numpy as np
import keras

class SEM_loader:
    """
    This class loads SEM images from a given folder including subfolders
    with corresponding labels
    Functions:  Load data of given size in the format for a conventional CNN
                Load data of given size for a siamese network
    """

    def __init__(self,size,path):
        """
        Initializes data
        Input:
            lang:   which damages should be loaded
            size:   size of the images size[0] x size[1]
            path:   path to to the folder containing subfolders with images
        """

        self.lang = {   "Martensite":0,"Evolved":1,"Evovled":1,
                        "Interface":2, "Notch":3, "Boundary":4,
                        "Inclusion":5}

        compressedLang = []
        for key,value in self.lang.iteritems():
            compressedLang.append(value)
        compressedLang = set(compressedLang)
        self.NOC = len(compressedLang)

        self.data = {}

        IGNORE = ["/home/tom/Data/LabeledDamages/CFK_def13_rep_1_2017-08-31"]

        ##
        ## Get subfolders
        ##

        subfolders = [x[0] for x in os.walk(path)]
        subfolders = subfolders[1:]

        for ign in IGNORE:
            if ign in subfolders:
                subfolders.remove(ign)
        # subfolders = [IGNORE]

        x = []
        y = []

        for folder in subfolders:
            print "Reading", folder
            for PATH_XML in os.listdir(folder):
                if PATH_XML.endswith(".xml"):
                    PATH_IMG = folder + "/" + PATH_XML[0:-4] + ".png"
                    img = misc.imread(PATH_IMG,flatten=True)

                    (ymax,xmax) = img.shape[0], img.shape[1]

                    assert (size[0]<xmax and size[1]<ymax), "The dimensions desired for the damage are to big"

                    tree = ET.parse(folder+"/"+PATH_XML)
                    root = tree.getroot()

                    for ob in root:
                        if ob.tag == "object":
                            ##
                            ## Check if the object is of interest
                            ##
                            INTEREST = True
                            for bbox in ob:
                                if bbox.tag == "name":
                                    if not bbox.text in self.lang:
                                        INTEREST = False
                                        break
                                    else:
                                        category = bbox.text
                                if bbox.tag == "bndbox":
                                    x1 = int(bbox[0].text)
                                    y1 = int(bbox[1].text)
                                    x2 = int(bbox[2].text)
                                    y2 = int(bbox[3].text)

                                    centerx = (x1+x2)/2
                                    centery = (y1+y2)/2

                                    ##
                                    ## Calculate the position of the damage in the image
                                    ##

                                    x1 = centerx - size[0]/2
                                    y1 = centery - size[1]/2
                                    x2 = centerx + size[0]/2
                                    y2 = centery + size[1]/2

                                    ##
                                    ## Catch the cases in which the extract would go
                                    ## over the boundaries of the original image
                                    ##

                                    if x1<0:
                                        x1 = 0
                                        x2 = size[0]
                                    if x2>=xmax:
                                        x1 = xmax - size[0]
                                        x2 = xmax
                                    if y1<0:
                                        y1 = 0
                                        y2 = size[1]
                                    if y2>=ymax:
                                        y1 = ymax - size[1]
                                        y2 = ymax

                            if INTEREST:
                                tmp = np.zeros((size[1],size[0],1))
                                tmp[:,:,0] = img[y1:y2,x1:x2]
                                tmp = tmp*2./255. - 1.
                                x.append(tmp)
                                y.append(self.lang[category])

        for key, value in self.lang.iteritems():
            print key, ": ", y.count(value)

        print "Size of the data set: ", len(y)

        x = np.asarray(x, float)
        y = np.asarray(y, int)

        # y = keras.utils.to_categorical(y, self.NOC)
        # y = keras.utils.to_categorical(y, len(lang))

        self.data = x
        self.label = y
        self.shape = size
        self.N = y.shape[0]

        self.lang = {   "Martensite":0,"Evolved":1, "Interface":2, "Notch":3, "Boundary":4,
                        "Inclusion":5}
        self.inv_lang = {v: k for k,v in self.lang.iteritems()}

    def getData(self, lang, split, dist=None):
        """
        Get the data and split it into training and test sets
        Input:
            lang:   which labels to use
            split:  split ratio between training and test sets
            dist:   how many examples per class should be used
        """

        ##
        ## Number of examples
        ##

        compressedLang = []
        for key,value in lang.iteritems():
            compressedLang.append(value)
        compressedLang = set(compressedLang)
        NOC = len(compressedLang)

        inv_lang = {v: k for k,v in lang.iteritems()}

        if dist!=None:

            x_tmp = []
            y_tmp = []

            D = {k:0 for k,v in dist.iteritems()}   # Distribution at the moment
            A = 0                                   # Number of data added
            S = [v for k,v in dist.iteritems()]
            S = sum(S)
            B = 0                                   # Break parameter to prevent
                                                    # entering a non ending loop

            while A<S and B<self.N:
                ##
                ## Check if the label y[B] is of interest
                ##
                if self.inv_lang[self.label[B]] in lang:
                    ##
                    ## Translate the label to using the new dictionary lang
                    ##
                    label = lang[self.inv_lang[self.label[B]]]
                    if D[self.inv_lang[self.label[B]]] < dist[self.inv_lang[self.label[B]]]:
                        x_tmp.append(self.data[B])
                        y_tmp.append(label)
                        D[self.inv_lang[self.label[B]]] += 1
                        A += 1
                B += 1

            for key, value in lang.iteritems():
                print key, ": ", y_tmp.count(value)

            x_tmp = np.asarray(x_tmp, float)
            y_tmp = np.asarray(y_tmp, int)
            y_tmp = keras.utils.to_categorical(y_tmp, NOC)

            ##
            ## Shuffle the data
            ##
            perm = np.random.permutation(range(A))
            x_tmp = x_tmp[perm]
            y_tmp = y_tmp[perm]

            ##
            ## Split data into training and testing data sets
            ##
            x_train = x_tmp[0:int((1-split)*A)]
            y_train = y_tmp[0:int((1-split)*A)]
            x_test = x_tmp[int((1-split)*A):A]
            y_test = y_tmp[int((1-split)*A):A]

            print x_train.shape
            print y_train.shape
            print x_test.shape
            print y_test.shape

        else:
            A = 0
            for i in range(self.N):
                ##
                ## Check if the label y[B] is of interest
                ##
                if self.inv_lang[self.label[i]] in lang:
                    label = lang[self.inv_lang[self.label[i]]]
                    x_tmp.append(self.data[i])
                    y_tmp.append(label)
                    A += 1

            x_tmp = np.asarray(x_tmp, float)
            y_tmp = np.asarray(y_tmp, int)

            y_tmp = keras.utils.to_categorical(y_tmp, NOC)

            ##
            ## Shuffle the data
            ##
            perm = np.random.permutation(range(A))
            x_tmp = x_tmp[perm]
            y_tmp = y_tmp[perm]

            ##
            ## Split data into training and testing data sets
            ##
            x_train = x_tmp[0:int((1-split)*A)]
            y_train = y_tmp[0:int((1-split)*A)]
            x_test = x_tmp[int((1-split)*A):A]
            y_test = y_tmp[int((1-split)*A):A]

        return x_train,y_train,x_test,y_test

    def getShape(self):
        return self.shape

    def getLang(self):
        return self.lang
