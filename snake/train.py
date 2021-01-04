from environment import Environment
from brain import Brain
from dqn import Dqn
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.0001
max_memory = 60000
gamma = 0.9
batch_size = 32
last_states = 4

epsilon = 1.0
epsilon_decay_rate = 0.0002
min_epsilon = 0.7

filepath_to_save = 'C:/Users/Saathvik/PycharmProjects/rl/snake/model.h5'

env = Environment(0)
brain = Brain((env.nColumns, env.nRows, last_states))
brain.model = brain.load_model(filepath_to_save)
model = brain.model
dqn = Dqn(max_memory, gamma)

def reset_states():
    current_state = np.zeros((1, env.nColumns, env.nRows, last_states))
    for i in range(last_states):
        current_state[0, :, :, i] = env.screenMap

    return current_state, current_state

epoch = 0
collected = 0
max_collected = 0
total_collected = 0
j = 0
scores = []
for i in range(10000):
    epoch += 1

    env.reset(plot = True)
    current_state, next_state  = reset_states()
    game_over = False
    while not game_over:
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 4)
        else:
            qvalues = model.predict(current_state)[0]
            action = np.argmax(qvalues)

            frame, reward, game_over = env.step(action, plot = True)
            frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
            next_state = np.append(next_state, frame, axis = 3)
            next_state = np.delete(next_state, 0, axis=3)

            dqn.remember([current_state, action, reward, next_state], game_over)
            inputs, outputs = dqn.get_batch(model, batch_size)
            model.train_on_batch(inputs, outputs)

            if env.collected:
                collected +=1

            current_state = next_state

        epsilon -= epsilon_decay_rate
        epsilon = max(epsilon, min_epsilon)

        if collected > max_collected:
            max_collected = collected

        total_collected += collected
        collected = 0

        print('\r epoch: ' + str( epoch ) + ' current max apples: ' + str( max_collected ) + ' epsilon: {:.5f}'.format(epsilon), end = '')
    print('\repoch: ' + str(epoch) + ' current max apples: ' + str(max_collected) + ' epsilon: {:.5f}'.format(epsilon), ' collected: ' + str(collected), ' total_collected: ', total_collected)
    j += 1
    if j == 10:
        model.save( filepath_to_save )
        j = 0

model.save(filepath_to_save)

scores.append( total_collected / 100 )
total_collected = 0
plt.plot( scores )
plt.xlabel( 'epoch / 100' )
plt.ylabel( 'average collected' )
plt.show()
plt.savefig( str( epoch ) )



