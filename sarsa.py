import numpy as np
import math
import gym

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here

        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        x_tile_count = math.ceil((state_high[0] - state_low[0]) / tile_width[0]) + 1
        y_tile_count = math.ceil((state_high[1] - state_low[1]) / tile_width[1]) + 1

        self.x_tile_count = x_tile_count
        self.y_tile_count = y_tile_count



    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """

        # TODO: implement this method

        d = self.num_actions * self.num_tilings * self.x_tile_count * self.y_tile_count
        return d

        raise NotImplementedError()

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method

        features = np.zeros((self.x_tile_count, self.y_tile_count, self.num_actions, self.num_tilings))

        if done is not True:
            for i in range(self.num_tilings):
                xtile = math.trunc(abs(s[0] - (self.state_low[0] - i / self.num_tilings * self.tile_width[0])) / self.tile_width[0])
                ytile = math.trunc(abs(s[1] - (self.state_low[1] - i / self.num_tilings * self.tile_width[1])) / self.tile_width[1])
                if xtile < 6 and ytile < 6:
                    features[xtile, ytile, a, i] = 1

        features = features.flatten()

        return features

        raise NotImplementedError()

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function

    for i in range(num_episode):

        state = env.reset()

        action = epsilon_greedy_policy(state, False, w, .1)

        x = X(state, False, action)

        z = np.zeros((X.feature_vector_len()))

        Q_old = 0

        done = False

        while not done:

            next_state, reward, done, info = env.step(action)

            # print(next_state)

            next_action = epsilon_greedy_policy(next_state, done, w, .1)

            x_prime = X(next_state, done, next_action)

            Q = np.dot(w, x)

            Q_prime = np.dot(w, x_prime)

            delta = reward + gamma*Q_prime - Q

            z = gamma*lam*z + (1 - alpha*gamma*lam*np.dot(z, x))*x

            w = w + alpha*(delta + Q - Q_old)*z - alpha*(Q - Q_old)*x

            Q_old = Q_prime

            x = x_prime

            action = next_action

    return w

    raise NotImplementedError()
