import gym

env = gym.make("CartPole-v0")
for i_episode in range(20):
    observation = env.reset()
    for t in range(10000):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        #print(action)
        action = [0]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after " + str(t+1) + " timesteps")
            break
