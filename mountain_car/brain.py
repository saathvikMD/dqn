import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Brain():
    def __init__(self, num_inputs, num_outputs, lr):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
        self.model = Sequential()
        self.model.add(Dense(32, activation = 'relu', input_shape = (self.num_inputs,  )))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.num_outputs))
        self.model.compile(optimizer = Adam(lr=self.lr), loss = 'mean_squared_error')