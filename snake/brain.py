import keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.optimizers import Adam

class Brain():
    def __init__(self, input_shape, lr = 0.005):
        self.input_shape = input_shape
        self.lr = lr
        self.num_outputs = 4

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape = self.input_shape))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(64, (2, 2), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.num_outputs))
        self.model.compile(optimizer=Adam(lr = self.lr), loss = 'mean_squared_error', metrics = ['accuracy'])

    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model