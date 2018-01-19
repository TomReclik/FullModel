import sys
sys.path.append("/home/tom/FullModel/")
import os
import unittest
import json
from models.classifier import combined_classifier
from utils.Localizer import Localizer

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.settings = {"ClassifierArchitecture":[
          {"Model":"InceptionV3",
           "Classes":{"Inclusion":0, "Rest":1},
           "WindowSize": [250,250],
           "weights": "/home/tom/FullModel/models/InceptionV3_IncVsAll_NoInc.hdf5"},
          {"Model":"EERACN",
           "Classes":{"Martensite":0,"Interface":1,"Notch":2,"Boundary":3},
           "WindowSize": [50,50],
           "weights": "/home/tom/FullModel/models/EERACN_SecondStage.hdf5"}
         ],
         "Classes":["Inclusion","Martensite","Interface","Notch","Boundary"],
         "Training":[
           {"dist":{"Inclusion":383, 
                    "Notch":100, 
                    "Interface":100, 
                    "Martensite":100, 
                    "Boundary":100},
            "class_weight": {0:1,1:1}},
            {"dist":{"Notch":443, 
                     "Interface":788, 
                     "Martensite":692, 
                     "Boundary":115},
             "class_weight": {0:1,1:1,2:1,3:1}}
         ],
         "TrainingDataFolder": "/home/tom/Data/LabeledDamages/",
         "Prediction":{
           "thresholds": [0.7,0.7]
         }
        }
        self.settingsjson = "/home/tom/FullModel/settings_test.json"
        with open(self.settingsjson, 'w') as fp:
            json.dump(self.settings, fp)
        self.cc = combined_classifier(self.settingsjson)
        
    def test_initialization_of_combined_classifier(self):
        assert self.cc.model == ["InceptionV3","EERACN"], "Models not correctly initialized"
        assert self.cc.window_size == [[250,250],[50,50]], "Window size not correctly initialized"
        assert self.cc.lang[0] == {"Inclusion":0, "Martensite":1, "Interface":1, "Notch":1, "Boundary":1}, self.cc.lang[0]
        assert self.cc.lang[1] == {"Martensite":0, "Interface":1, "Notch":2, "Boundary":3}, self.cc.lang[1]

    def tearDown(self):
        os.remove(self.settingsjson)

if __name__ == "__main__":
    unittest.main()
    
