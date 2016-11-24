import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
gamma = 0.99
alpha = 0.00001
action = 1
w = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
delta_w = np.array([0,0,0,0,0,0,0,0,0,0,0])

def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle

def state_action_to_features(state, action):
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    X = np.array([normalize_angle(theta), theta % (2*np.pi),
            normalize_angle(phi), phi % (2*np.pi),
            abs(theta_dot), theta_dot,
            abs(phi_dot), phi_dot,
            action*theta, action*phi,
            action*theta_dot, action*phi_dot])
    return X

def q_hat(state, action, w):
    X = state_action_to_features(state, action)
    output = np.dot(X,w)
    return output
    
# epislon-greedy:
def get_action(state, w):
    actions = [-40, 40]
    qs = []
    for action in actions:
        qs.append(q_hat(state, action, w))
    max_index = np.argmax(qs)
    action = actions[max_index]
    return action

timesteps = []
for i_episode in range(1000):
    state = env.reset()
    action = get_action(state, w)
    for t in range(100000):
        env.render()
        action = get_action(state, w)
        #print(action)
        observation, reward, done, info = env.step(action)
        # update w
        delta_w = (alpha*(reward + gamma*q_hat(observation, get_action(observation, w), w) - q_hat(state, action, w)))*state_action_to_features(state, action)
        w = np.add(w,delta_w)
        
        print(w)
        #print(reward)
        state = observation
        
        if done:
            print("Episode finished after " + str(t+1) + " timesteps")
            timesteps += [t+1]
            break
plt.plot(timesteps)
plt.show()
