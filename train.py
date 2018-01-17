from models.classifier import combined_classifier

settingsjson = "/home/tom/FullModel/settings.json"
cc = combined_classifier(settingsjson)
cc.train(settingsjson)
