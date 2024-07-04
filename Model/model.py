import logging
import torch
import os
import numpy as np

class RecognitionModel:
    def __init__(self, path, args):
        logging.info("Loading models...")
        print("Model path", path[0])
        print("Weights path", os.path.join(path[1], args.exp, "weights", "last.pt"))
        self.Sign_model = torch.hub.load(path[0], 'custom', source='local', path=os.path.join(path[1], args.exp, "weights", "last.pt"), force_reload=True, _verbose=False)
        logging.info("Model loaded successfully.")

    def FRSign_recognition(self, img):
        detection = self.Sign_model(img)
        #Convert detection into data frame
        data_frame = detection.pandas().xyxy[0]
        #Extract indexes of data frame
        indexes = data_frame.index
        #Initialize the Results array with correct shape
        Results = np.zeros((len(indexes), 6))
        for i in indexes:
            #Get X_min, Y_min, X_max & Y_max
            x_min = int(data_frame['xmin'][i])
            y_min = int(data_frame['ymin'][i])
            x_max = int(data_frame['xmax'][i])
            y_max = int(data_frame['ymax'][i])
            #Get label & confidence
            name_class = int(data_frame['class'][i])
            confidence = float(data_frame['confidence'][i]) * 100
            #Store information
            Results[i, :] = (x_min, y_min, x_max, y_max, confidence, name_class)
        
        #information is sorted according to xmin values
        sorted_indexes = np.argsort(Results[:, 0])
        Results_sorted = Results[sorted_indexes]
        
        #rows with low confidence are removed
        Results_Filtered = self.threshold(Results_sorted)
        return Results_Filtered
    
    def threshold(self, Array):
        filtered_array = Array[Array[:, 4] >= 45.0]
        return filtered_array