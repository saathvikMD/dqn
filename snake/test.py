from time import sleep
from environment import Environment
from brain import Brain
import numpy as np

waitTime = 75
last_states = 4

filepath_to_open = 'C:/Users/Saathvik/PycharmProjects/rl/snake/model.h5'

env = Environment(waitTime)
brain = Brain((env.nColumns, env.nRows, last_states))
model = brain.load_model(filepath_to_open)


def reset_states():
    current_state = np.zeros((1, env.nColumns, env.nRows, last_states))
    for i in range(last_states):
        current_state[0, :, :, i] = env.screenMap

    return current_state, current_state

while True:
    env.reset(plot = True)
    sleep(10)
    current_state, next_state = reset_states()
    game_over = False
    while not game_over:
        qvalues = model.predict(current_state)[0]
        action = np.argmax(qvalues)
        frame, _, game_over = env.step( action, plot=True )
        frame = np.reshape( frame, (1, env.nColumns, env.nRows, 1) )
        next_state = np.append( next_state, frame, axis=3 )
        next_state = np.delete( next_state, 0, axis=3 )

        current_state = next_state
