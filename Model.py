from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from PIL import Image

class NeuralNetwork():
    mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,50),learning_rate='adaptive') #best acc after grid search
    X=None #2d matrix of features 
    y=None #1d array of targets

    def loadData784(self) -> None:
        print("Loading data from file...")
        try:
            rawData= loadarff('mnist_784.arff')
        except FileNotFoundError:#in case the file is not downloaded
            rawData=fetch_openml('mnist_784', version=1, return_X_y=True)
        df= pd.DataFrame(rawData[0])
        self.y=df['class'].astype(int).values #masking to get only target values
        self.X=df.drop(columns='class').astype(int).values #all except target values
        print("Loading complete!")

    def train(self):
        print("Training model... (Might take a couple of minutes)")
        self.mlp.fit(self.X,self.y)
        print("Training complete!")
        print(f"Accuracy on trained data: {self.mlp.score(self.X,self.y)}")

    def predictFromImage(self):
        looping = True
        while looping:
            filename = str(input("Make a 28x28 image and type its name with extension: [number.png]"))
            if filename=="":
                img =Image.open('number.png')
            else:
                img = Image.open(filename)
            imgGrayscale=img.convert('L')
            imgData=np.array(imgGrayscale).reshape(-1)
            print("Prediction: ",self.mlp.predict(imgData.reshape(1,-1)))
            again=str(input("Predict again? y/n [y]"))
            if again=="n":
                looping=False
        print("Predicting ended.")
