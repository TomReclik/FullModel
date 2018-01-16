from models.classifier import classifier

weights_path = "/home/tom/FullModel/models/InceptionV3_IncVsAll_NoInc.hdf5"
model = "InceptionV3"
window_size = [250,250]

FirstStage = classifier(model, window_size)

folder="/home/tom/Data/LabeledDamages/"
lang = {"Inclusion":0, "Notch":1, "Interface":1, "Martensite":1, "Boundary":1}
dist = {"Inclusion":383, "Notch":100, "Interface":100, "Martensite":100, "Boundary":100}
weights_path = "/home/tom/FullModel/models/FirstStage.hdf5"

FirstStage.train(folder,lang,weights_path,batch_size=10,epochs=150,
          validation_split=0.2,class_weight=None,dist=dist)
