import sys

sys.path.append("/home/tom/FullModel/")

from models.classifier import combined_classifier
cc = combined_classifier("/home/tom/FullModel/settings.json")

print cc.model
