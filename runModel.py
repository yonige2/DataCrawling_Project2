from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np

class RunModel:
    def loadModel(self, path):
        model = keras.models.load_model(path)
        return model


    def preprocessData(self, data):
        column = len(data)
        row = len(data[0])

        X = np.array(data)
        X = X.reshape(1, column, row, 1)
        
        return X


    def runModel(self, model, X, minNumber, maxNumber):
        result = model.predict(X)
        result = ((maxNumber - minNumber) * (result)) + minNumber

        return int(result)