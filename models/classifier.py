import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import InceptionV3
import models
from utils.utils import SEM_loader
import numpy as np
import json

class classifier:
    def __init__(self, model, NOC, window_size, weights_path=None):
        self.models = ["InceptionV3", "EERACN"]

        if not model in self.models:
            raise "Specified model is not implemented"

        if model == "InceptionV3":
            self.NOC = NOC
            self.network = InceptionV3.InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=[250,250,1],
                            pooling=None,
                            classes=self.NOC)
        elif model == "EERACN":
            self.NOC = NOC
            self.network = models.EERACN([window_size[0],window_size[1],1], self.NOC)

        if weights_path!=None:
            self.network.load_weights(weights_path)
        self.window_size = window_size

    def train(self,folder,lang,weights_path,batch_size=10,epochs=150,
              validation_split=0.2,class_weight=None,dist=None):

        compressedLang = []
        for key,value in lang.iteritems():
            compressedLang.append(value)
        compressedLang = set(compressedLang)
        NOC = len(compressedLang)

        if class_weight==None:
            class_weight = {}
            for i in range(NOC):
                class_weight[i] = 1

        print "Loading Data"

        loader = SEM_loader(self.window_size,folder)

        x_train, y_train, x_test, y_test = loader.getData(lang,0.2,dist=dist)

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, verbose=0),
                ModelCheckpoint(weights_path, monitor='val_loss', verbose=0,
                save_best_only=True, save_weights_only=False, mode='auto', period=1)
            ]
        self.network.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_split=0.2, class_weight=class_weight)

        score = self.network.evaluate(x_test, y_test, batch_size=batch_size)

        print score

    def predict(self, SEM, centroids):
        """
        Given SEM micrograph together with the centroids of the predicted
        damage sites classify them
        Input:
            SEM: Gray scale image format: (N,M)
            centroids: centers of the predicted damage site locations
        """
        size = SEM.shape

        x= []

        for i in range(len(centroids)):
            x1 = int(centroids[i][0] - self.window_size[0]/2)
            y1 = int(centroids[i][1] - self.window_size[1]/2)
            x2 = int(centroids[i][0] + self.window_size[0]/2)
            y2 = int(centroids[i][1] + self.window_size[1]/2)

            ##
            ## Catch the cases in which the extract would go
            ## over the boundaries of the original image
            ##

            if x1<0:
                x1 = 0
                x2 = self.window_size[0]
            if x2>=size[0]:
                x1 = size[0] - self.window_size[0]
                x2 = size[0]
            if y1<0:
                y1 = 0
                y2 = self.window_size[1]
            if y2>=size[1]:
                y1 = size[1] - self.window_size[1]
                y2 = size[1]

            tmp = np.zeros((self.window_size[1],self.window_size[0],1), dtype=float)
            # print SEM[x1:x2,y1:y2].shape
            tmp[:,:,0] = SEM[x1:x2,y1:y2]
            tmp = tmp*2./255. - 1.
            x.append(tmp)

        x = np.asarray(x, float)

        y_pred = self.network.predict(x)

        return y_pred

class combined_classifier:
    def __init__(self,settingsjson):
        """
        Initializes combined network from a given json file. The supported
        architectural design is to filter out classes from top to bottom.
        Different classifiers on the same level are not possible. E.g. We have
        the classes "Inclusion", "Martensite", "Interface Decohesion". In the 
        first step we want to distinguish between "Inclusions" and the rest.
        This is done using an InceptionV3 network. In the next step all sites
        that aren't "Inclusion" will be given to the next classifier.
        """
        with open(settingsjson, 'r') as f:
            settings = json.load(f)
            
        self.settings = settings
        self.networks = []
        self.NOC = []
        self.lang = []
        self.window_size = []
        self.model = []
        self.all_classes = settings["Classes"]
        remaining_classes = settings["Classes"]
        self.weights_path = []
        for i in range(len(settings["ClassifierArchitecture"])):
            
            if "weights" in settings["ClassifierArchitecture"][i].keys():
                weights_path = settings["ClassifierArchitecture"][i]["weights"]
            else:
                weights_path = None
            
            window_size = settings["ClassifierArchitecture"][i]["WindowSize"]
            model = settings["ClassifierArchitecture"][i]["Model"]
            lang = settings["ClassifierArchitecture"][i]["Classes"]
            NOC = len(lang)
            
            ##
            ## If the classifier is not the last one all classes that are classified
            ## by this classifier will be removed from all_classes_tmp, such
            ## that following classes will not try to reclassify preceding
            ## classes
            ##
            
            for key in lang.keys():
                if(key!="Rest"):
                    remaining_classes.remove(key)
                
            ##
            ## If the classifier is not the last one it will contain the class
            ## "Rest" being a placeholder for all classes not classified yet
            ## "Rest" will then be removed and replaced by all remaining classes
            ## with its value being the maximal value of lang + 1
            ##
            
            if "Rest" in lang.keys():
                del lang["Rest"]
                max_key = max(lang.values())
                for c in remaining_classes:
                    lang[c] = max_key + 1
            
            network = classifier(model, NOC, window_size, weights_path=weights_path)
            
            self.model.append(model)
            self.window_size.append(window_size)
            self.networks.append(network)
            self.NOC.append(NOC)
            self.lang.append(lang)
            self.weights_path.append(weights_path)
            
    def train(self):
        """
        Trains the networks seperately on the given data.
        In this case folder specifies the superfolder with the data together
        with the annotations in xml format in the subfolders
        """
        folder = self.settings["TrainingDataFolder"]
        for i in range(len(self.settings["Training"])):
            print "Training ", i+1, " Network"
            dist = self.settings["Training"][i]["dist"]
            class_weight = self.settings["Training"][i]["class_weight"]
            
            weights_path = "/home/tom/FullModel/models/"  + self.model[i] + "_" + str(i) + ".hdf5"
            self.weights_path.append(weights_path)
            self.networks[i].train(folder,self.lang[i],weights_path,batch_size=10,epochs=150,
                      validation_split=0.2,class_weight=class_weight,dist=dist)
                      
    def predict(self, SEM, centroids):
        """
        Using the initialized network architecture predict the classes in the 
        SEM micrograph
        """
        
        annotations = ["Not Classified" for i in range(len(centroids))]
        
        thresholds = self.settings["Prediction"]["thresholds"]
        centroids_tmp = centroids.tolist()
        not_assigned = range(len(centroids_tmp))
        
        for i in range(len(self.networks)):
            y_pred = self.networks[i].predict(SEM, centroids_tmp)
            inv_lang = {v: k for k,v in self.lang[i].iteritems()}
        
            ##
            ## If we aren't at the last network and none of the classes return 
            ## a probability greater than the threshold value the image will 
            ## be passed on to the next network. Therefore we remove the last 
            ## column of y_pred if we are not in the last network 
            ##
            
            if i!=len(self.networks)-1:
                y_shape = y_pred.shape
                y_pred = y_pred[:,0:y_shape[1]-1]
                
                
            c = np.where(y_pred>thresholds[i])
            
            ##
            ## c[0][j] corresponds to the centroid being processed
            ## c[1][j] corresponds to the class of the image
            ##
            assigned = []
            assigned_centroids = []
            
            for j in range(c[0].shape[0]):
                annotations[not_assigned[c[0][j]]] = inv_lang[c[1][j]]
                assigned.append(not_assigned[c[0][j]])
                assigned_centroids.append(centroids_tmp[c[0][j]])
                
            ##
            ## Remove already assigned images
            ##
            
            for a in assigned:
                not_assigned.remove(a)
            for ac in assigned_centroids:    
                centroids_tmp.remove(ac)
                            
        return annotations
        
            























            
