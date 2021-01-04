import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from environment import Gridworld

def make_noise(min = 1, max = 64, noise = 10):
    return np.random.rand(min, max) / noise

env = Gridworld(size=4, mode='static')

class Dqn():
    def __init__(self, max_memory, input_shape, output_shape, lr):
        self.lr = lr
        self.memory = []
        self.discount = 0.9
        self.max_memory = max_memory
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.input_shape, input_shape = (1, self.input_shape)))
        self.model.add(keras.layers.Dense(self.input_shape * 2, activation = 'relu'))
        self.model.add(keras.layers.Dense(self.input_shape * 3))
        self.model.add(keras.layers.Dense(self.input_shape * 2, activation = 'relu'))
        self.model.add( keras.layers.Dense( self.input_shape //2, activation='relu' ) )
        self.model.add(keras.layers.Dense(self.output_shape))
        self.model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate=self.lr), metrics = ['accuracy'])

    def remember(self, current_state, action, reward, next_state, game_over):
        transition = [current_state, action, reward, next_state]
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
          del self.memory[0]

    def get_batch(self, batch_size, model):
        len_memory = len(self.memory)
        num_inputs = 64
        num_outputs = 4

        inputs = np.zeros((min(batch_size, len_memory), num_inputs))
        targets = np.zeros((min(batch_size, len_memory), num_outputs))
        for i, inx in enumerate(np.random.randint(0, len_memory, size = min(batch_size, len_memory))):
            inx = -1
            current_state, action, reward, next_state = self.memory[inx][0]
            game_over = self.memory[inx][1]

            inputs[i] = current_state
            targets[i] = model.predict(np.array(current_state).reshape(1, 1, self.input_shape))[0]

            if game_over:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.discount * np.max(model.predict(np.array(next_state).reshape(1, 1, self.input_shape))[0])

        return inputs.reshape(min(len_memory, batch_size), 1, self.input_shape), targets.reshape(min(len_memory, batch_size), 1, self.output_shape)


    def train(self, input, output, verbose = 1):
        history = self.model.fit(input, output, epochs=1, verbose = verbose)
        return history.history['loss'][-1], history.history['accuracy'][-1]

def calculate_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

dqn = Dqn(1000, 64, 4, 0.001)
loss = []
step_loss = []
st = env.board.render_np().reshape(1, 1, 64)
state = st
action_set = {0:'u', 1:'d', 2:'l', 3:'r'}

epsilon = 1.0
epsilon_decay_rate = 0.995
done = False
max_moves = 50

target_model = keras.models.clone_model(dqn.model)
model_update = 500
o = 0

for i in range(1000):
    losses = []
    game = Gridworld(mode='random')
    epsilon = 1.0
    total_reward = 0
    done = False
    j = 0
    while not done and j < max_moves:
        action = np.argmax(dqn.model.predict(st)[0])
        if random.random() > epsilon:
            action = random.randint(0, 3)
        game.makeMove(action_set[action])
        epsilon = epsilon * epsilon_decay_rate
        reward = game.reward()
        done = not reward == -1
        state = game.board.render_np().reshape(1, 1, 64)
        dqn.remember(st, action, reward, state, done)
        input, output = dqn.get_batch(200, target_model)
        l, a = dqn.train(input, output, verbose = 0)
        losses.append(l)
        total_reward += reward
        st = state
        j += 1
        o += 1
        if o == model_update:
            target_model = keras.models.clone_model( dqn.model )
            o = 0

    step_loss.append(losses)
    print('epoch:', str(i + 1), total_reward, j, total_reward == (-j + 11), o)
    loss.append(total_reward)

plt.plot(loss)