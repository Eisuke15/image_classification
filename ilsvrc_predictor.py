import json
import numpy as np

ILSVRC_calss_index = json.load(open('data/imagenet_class_index.json', 'r'))

class ILSVRCPredictor:

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]
        return predicted_label_name
