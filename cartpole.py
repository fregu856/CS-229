"""
Original file: cartpole.py
Modifications by Fredrik Gustafsson
"""


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.g = 9.81 # gravity constant
        self.m0 = 1.0 # mass of cart
        self.m1 = 0.1 # mass of pole 1
        self.m2 = 0.1 # mass of pole 2
        self.L1 = 1 # length of pole 1
        self.L2 = 1 # length of pole 2
        self.l1 = self.L1/2 # distance from pivot point to center of mass
        self.l2 = self.L2/2 # distance from pivot point to center of mass
        self.I1 = self.m1*(self.L1^2)/12 # moment of inertia of pole 1 w.r.t its center of mass
        self.I2 = self.m2*(self.L2^2)/12 # moment of inertia of pole 2 w.r.t its center of mass
        self.tau = 0.01  # seconds between state updates

        # Angle at which to fail the episode
        #self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # # (never fail the episode based on the angle)
        self.theta_threshold_radians = 100000 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        #self.action_space = spaces.Discrete(2)
        # # (continuous action space)
        self.action_space = spaces.Box(low=-100, high=100, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot, phi, phi_dot = state
        u = action
        
        d1 = self.m0 + self.m1 + self.m2
        d2 = self.m1*self.l1 + self.m2*self.L1
        d3 = self.m2*self.l2
        d4 = self.m1*pow(self.l1,2) + self.m2*pow(self.L1,2) + self.I1
        d5 = self.m2*self.L1*self.l2
        d6 = self.m2*pow(self.l2,2) + self.I2
        f1 = (self.m1*self.l1 + self.m2*self.L1)*self.g 
        f2 = self.m2*self.l2*self.g    
        
        D = np.matrix([[d1, d2*math.cos(theta), d3*math.cos(phi)], 
                [d2*math.cos(theta), d4, d5*math.cos(theta-phi)],
                [d3*math.cos(phi), d5*math.cos(theta-phi), d6]])
        
        C = np.matrix([[0, -d2*math.sin(theta)*theta_dot, -d3*math.sin(phi)*phi_dot],
                [0, 0, d5*math.sin(theta-phi)*phi_dot],
                [0, -d5*math.sin(theta-phi)*theta_dot, 0]])
                
        G = np.matrix([[0], [-f1*math.sin(theta)], [-f2*math.sin(phi)]])
        
        H  = np.matrix([[1],[0],[0]])
        
        I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        O_3_1 = np.matrix([[0], [0], [0]])
        
        A_tilde = np.bmat([[O_3_3, I],[O_3_3, -np.linalg.inv(D)*C]])
        B_tilde = np.bmat([[O_3_1],[np.linalg.inv(D)*H]])
        W = np.bmat([[O_3_1],[np.linalg.inv(D)*G]])
                
                
        
        
        
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        #screen_width = 600
        #screen_height = 400
        # #
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        #carty = 100 # TOP OF CART
        # #
        carty = 300 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.8
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole2.set_color(.2,.6,.4)
            self.poletrans2 = rendering.Transform(translation=(0, polelen-5))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.poletrans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            self.axle2 = rendering.make_circle(polewidth/2)
            self.axle2.add_attr(self.poletrans2)
            self.axle2.add_attr(self.poletrans)
            self.axle2.add_attr(self.carttrans)
            self.axle2.set_color(.1,.5,.8)
            self.viewer.add_geom(self.axle2)
            
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        self.poletrans2.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
