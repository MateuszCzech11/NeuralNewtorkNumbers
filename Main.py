import Model
from PIL import Image
import numpy as np


network = Model.NeuralNetwork()
network.loadData784()
network.train()
network.predictFromImage()


