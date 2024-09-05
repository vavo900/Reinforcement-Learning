from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

        self.n_inputs = state_dims
        self.n_outputs = num_actions
        self.alpha = alpha

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs),
            nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.network.parameters(), alpha)

        self.loss = 0

    def __call__(self,s) -> int:
        # TODO: implement this method

        action_space = np.arange(self.n_outputs)
        action_probs = self.network(torch.FloatTensor(s))
        action_probs_2 = action_probs.detach().numpy()
        action = np.random.choice(action_space, p=action_probs_2)
        return action

        raise NotImplementedError()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method

        pass
        #raise NotImplementedError()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here

        self.n_inputs = state_dims
        self.n_outputs = 1
        self.alpha = alpha

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs),
            nn.Softmax(dim=-1))

    def __call__(self,s) -> float:
        # TODO: implement this method

        state_value = self.network(torch.FloatTensor(s))
        return state_value

        raise NotImplementedError()

    def update(self,s,G):
        # TODO: implement this method

        pass
        # raise NotImplementedError()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method

    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    batch_size = 1
    alpha = 0.01



    # action_space = np.arange(env.action_space.n)

    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        t = 0

        while not complete:

            t += 1
            # action_probs = pi(s_0).detach().numpy()
            action = pi(s_0)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            if complete:


                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                for i in range(t):

                    pi.optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    action_tensor = torch.LongTensor(batch_actions)

                    logprob = torch.log(pi.network(torch.FloatTensor(state_tensor)))
                    selected_logprobs = alpha*(gamma**t)*reward_tensor * \
                                        logprob[np.arange(len(action_tensor)), action_tensor]
                    pi.loss = -selected_logprobs.mean()

                    pi.loss.backward()
                    pi.optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

                print("total rewards", total_rewards)

    return total_rewards

    raise NotImplementedError()


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()