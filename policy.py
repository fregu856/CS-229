import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
gamma = 0.99
beta = 0.00001
alpha = 0.000001
sigma = 0.001
w = np.array([0, 0, 0, 0, 0, 0, 0, 0])
delta_w = np.array([0,0,0,0,0,0,0,0])
v = np.array([0,0,0,0,0,0,0,0])
delta_v = np.array([0,0,0,0,0,0,0,0])

def log(log_message):
    """
    
    DESCRIPTION:
    - Adds a log message "log_message" to a log file.
    
    """
    
    # open the log file and make sure that it's closed properly at the end of the 
    # block, even if an exception occurs:
    with open("C:/Users/Fregus/log2.txt", "a") as log_file:
        # write the log message to logfile:
        log_file.write(log_message)
        log_file.write("\n") # (so the next message is put on a new line)

def sign(x):
    if (x > 0):
        output = 1
    elif x < 0:
        output = -1
    else:
        output = 0
    return output

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

def phi_critic(state, action):
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    phi_critic = np.array([normalize_angle(theta),
            normalize_angle(phi), theta*theta_dot, phi*phi_dot,
            action*theta, action*phi,
            action*theta_dot, action*phi_dot])
    return phi_critic
    
def q_hat(state, action, w):
    X = phi_critic(state, action)
    output = np.dot(X,w)
    return output
    
def phi_actor(state):
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    ind1 = 0
    ind2 = 0
    if theta > 0 and phi < 0:
        ind1 = 1
    if theta < 0 and phi > 0:
        ind2 = 1
    phi_actor = np.array([theta, phi, theta_dot, phi_dot, sign(theta), sign(phi), ind1, ind2])
    return phi_actor
    
def mu(state, v):
    X = phi_actor(state)
    output = np.dot(X,v)
    return output
    
def get_action(state, v):
    mean = mu(s,v)
    action = np.random.normal(mean, sigma)
    if action > 40:
        action = 40
    elif action < -40:
        action = -40
    return action

while True:
    s = env.reset()
    a = get_action(s, v)
    for t in range(100000):
        #env.render()
        #print(a)
        s_, r, done, info = env.step(a)
        a_ = get_action(s_,v)
        
        # update v (actor):
        delta_v = alpha*(((a - mu(s, v))*phi_actor(s)))*q_hat(s, a, w)
        v = np.add(v, delta_v)
        
        # update w (critic):
        delta_w = (beta*(r + gamma*q_hat(s_, a_, w) - q_hat(s, a, w)))*phi_critic(s, a)
        w = np.add(w, delta_w)
        
        s = s_
        a = a_
        
        #print(v)
        
        if done:
            #print("Episode finished after " + str(t+1) + " timesteps")
            log(str(t+1))
            log(str(w))
            log(str(v))
            break

#[-6.46, -7.89, -8.74, -30.85, -30.07, -5.055,- 33.34, 23.09, 72.94, -38.84, 1.77, -55.54]
#[0.67428356, -0.06215373, 0.71389644, -0.04985835, -0.35220665, 0.91792232, 0.03106055, 0.78030967, 0.12260051, -0.18088769, 0.00512933, -0.07230826]

# w:
# [-39.69998563, -35.37277748, -137.50845676, -76.0650973, -111.99402429, -26.29163286, -87.21205602, 0.72850579]
# v:
# [39.6331613, 16.97342259, 111.29048407, -2.59649649, 45.7345003, -7.5349689, 2.80431484, -23.8304194]
