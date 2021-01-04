from dqn import Dqn
from brain import Brain

import gym
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
max_memory = 5000
gamma = 0.9
batch_size = 32
epsilon = 1.
epsilon_decay_rate = 0.995

env = gym.make('MountainCar-v0')
brain = Brain(2, 3, learning_rate)
model = brain.model
dqn = Dqn(max_memory, gamma)

epoch = 0
max_actions = 1000
current_state = np.zeros((1, 2))
next_state = current_state
total_reward = 0
rewards = []
while True:
    epoch +=1

    epoch += 1

    env.reset()
    current_state = np.zeros( (1, 2) )
    total_reward = 0
    game_over = False
    j = 0
    while not game_over and j < max_actions:
        if np.random.rand() <= epsilon:
            action = np.random.randint( 0, 3 )
        else:
            qvalues = model.predict( current_state )[0]
            action = action = np.argmax( qvalues )

        next_state[0], reward, game_over, info = env.step( action )

        total_reward += reward

        dqn.remember( [current_state, action, reward, next_state], game_over )
        inputs, targets = dqn.get_batch( model, batch_size )
        print( model.train_on_batch( inputs, targets ))

        current_state = next_state
        print( total_reward, next_state )

    epsilon *= epsilon_decay_rate
    print( 'Epoch: ', str( epoch ) + 'Epsilon: {:.5f}'.format( epsilon ) + 'Total Reward: {:.2f}'.format( total_reward ) )

    rewards.append( total_reward )
    total_reward = 0
    plt.plot( rewards )
    plt.xlabel( 'Epochs' )
    plt.ylabel( 'Rewards' )

env.close()