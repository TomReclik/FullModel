import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import InceptionV3
import models
from utils.utils import SEM_loader
import numpy as np

class classifier:
    def __init__(self, model, window_size, weights_path=None):
        self.models = ["InceptionV3", "EERACN"]

        if not model in self.models:
            raise "Specified model is not implemented"

        if model == "InceptionV3":
            self.NOC = 2
            self.network = InceptionV3.InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=[250,250,1],
                            pooling=None,
                            classes=self.NOC)
        elif model == "EERACN":
            self.NOC = 4
            self.network = models.EERACN([window_size[0],window_size[1],1], self.NOC)

        if weights_path!=None:
            self.network.load_weights(weights_path)
        self.window_size = window_size

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

class combined_classifier:
    def __init__(self,settingsjson):
        with open(settingsjson, 'r') as f:
            settings = json.load(f)

        self.networks = []
        self.classes = []
        for i in range(len(settings["Stages"])):
