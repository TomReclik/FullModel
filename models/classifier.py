import keras
import InceptionV3
import models
import numpy as np

class classifier:
    def __init__(self, weights_path, model, window_size):
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
            self.network.load_weights(weights_path)
            self.window_size = window_size
        elif model == "EERACN":
            self.NOC = 4
            self.network = models.EERACN([window_size[0],window_size[1],1], self.NOC)
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
