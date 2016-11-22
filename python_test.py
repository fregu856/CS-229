import gym
import numpy as np

env = gym.make("CartPole-v0")
gamma = 0.99
alpha = 2
action = 1
w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def state_action_to_features(state, action):
    X = [state[0][0]/(action+0.1), state[0][0]*action, state[1][0]*action, state[2][0]/(action+0.1), state[2][0]*action, -state[3][0]*action, state[4][0]*action, state[5][0]+action, 1, -action**2]
    return X

def q_hat(state, action, w):
    X = state_action_to_features(state, action)
    output = np.dot(X,w)
    return output
    
def get_action(state, w):
    actions = [-10,-8,-6,-4, -2, 0, 2, 4,6,8,10]
    qs = []
    for action in actions:
        qs.append(q_hat(state, action, w))
    max_index = np.argmax(qs)
    action = actions[max_index]
    return action

for i_episode in range(100):
    state = env.reset()
    for t in range(10000):
        env.render()
        action = get_action(state, w)
        print(action)
        observation, reward, done, info = env.step(action)
        
        # update w
        delta_w = alpha*(reward + gamma*q_hat(observation, get_action(observation, w), w) - q_hat(state, action, w))*state_action_to_features(state, action)
        w += delta_w

        state = observation
        if done:
            print("Episode finished after " + str(t+1) + " timesteps")
            break
