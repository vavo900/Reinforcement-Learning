from typing import Iterable
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


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
        self.learning_rate = alpha

        self.n_hidden_nodes = 32

        self.scope = "policy_estimator"

        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()

            self.state = tf.placeholder(tf.float32, [None, self.n_inputs],
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.actions = tf.placeholder(tf.int32, [None], name='actions')

            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)

            layer_2 = fully_connected(layer_1, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)

            output_layer = fully_connected(layer_2, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)

            self.action_probs = tf.squeeze(
                tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))

            indices = tf.range(0, tf.shape(output_layer)[0]) \
                      * tf.shape(output_layer)[1] + self.actions

            selected_action_prob = tf.gather(tf.reshape(self.action_probs, [-1]),
                                             indices)

            self.loss = -tf.reduce_mean(tf.log(selected_action_prob) * self.rewards)

            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32,
                                                           name='grads' + str(j)))

            self.gradients = tf.gradients(self.loss, self.tvars)

            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))

            self.init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init)

    def __call__(self,s) -> int:
        # TODO: implement this method
        s = np.reshape(s, (1, -1))
        probs = self.sess.run([self.action_probs],
                              feed_dict={
                                  self.state: s
                              })[0]

        action_space = np.arange(self.n_outputs)
        action = np.random.choice(action_space,
                                  p=probs)
        return action

        raise NotImplementedError()

    def update(self, gradient_buffer):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method

        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

        # raise NotImplementedError()

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, actions, rewards):
        grads = self.sess.run([self.gradients],
                              feed_dict={
                                  self.state: states,
                                  self.actions: actions,
                                  self.rewards: rewards
                              })[0]
        return grads



class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.scope = "Baseline"

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, rewards):
        grads = self.sess.run([self.gradients],
                              feed_dict={
                                  self.state: states,
                                  self.rewards: rewards
                              })[0]
        return grads

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
        self.learning_rate = alpha

        self.n_hidden_nodes = 32

        self.scope = "value_estimator"

        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()

            self.state = tf.placeholder(tf.float32, [None, self.n_inputs],
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)

            layer_2 = fully_connected(layer_1, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)

            output_layer = fully_connected(layer_2, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)

            self.state_value_estimation = tf.squeeze(output_layer)

            self.loss = tf.reduce_mean(tf.squared_difference(
                self.state_value_estimation, self.rewards))

            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32,
                                                           name='grads' + str(j)))

            self.gradients = tf.gradients(self.loss, self.tvars)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))

            self.init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init)

    def __call__(self,s) -> float:
        # TODO: implement this method

        s = np.reshape(s, (1, -1))

        value_est = self.sess.run([self.state_value_estimation],
                                  feed_dict={
                                      self.state: s
                                  })[0]
        return value_est

        raise NotImplementedError()

    def update(self,gradient_buffer):
        # TODO: implement this method

        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

        # raise NotImplementedError()

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, rewards):
        grads = self.sess.run([self.gradients],
                              feed_dict={
                                  self.state: states,
                                  self.rewards: rewards
                              })[0]
        return grads


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
    discounted_tot_rewards = []
    batch_size = 1

    grad_buffer_pe = pi.get_vars()
    for i, g in enumerate(grad_buffer_pe):
        grad_buffer_pe[i] = g * 0

    if isinstance(V, VApproximationWithNN):
        grad_buffer_ve = V.get_vars()
        for i, g in enumerate(grad_buffer_ve):
            grad_buffer_ve[i] = g * 0

    action_space = np.arange(env.action_space.n)

    for ep in range(num_episodes):

        s_0 = env.reset()
        reward = 0
        episode_log = []

        complete = False

        while complete == False:

            action = pi(
                s_0.reshape(1, -1))

            if isinstance(V, VApproximationWithNN):
                value_est = V(
                    s_0.reshape(1, -1))

            # action_probs = pi(
            #     s_0)
            #
            # if isinstance(V, VApproximationWithNN):
            #     value_est = V(
            #         s_0)

            # action = np.random.choice(action_space,
            #                           p=action_probs)

            s_1, r, complete, _ = env.step(action)

            if isinstance(V, VApproximationWithNN):
                re_delta = r - value_est
            else:
                re_delta = r - V(s_0)

            episode_log.append([s_0, action, re_delta, r, s_1])
            s_0 = s_1

            if complete:
                episode_log = np.array(episode_log)

                total_rewards.append(episode_log[:, 3].sum())
                discounted_rewards = discount_rewards(
                    episode_log[:, 3], gamma)
                discounted_reward_est = discount_rewards(
                    episode_log[:, 2], gamma)
                discounted_tot_rewards.append(discounted_rewards.sum())

                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

                pe_grads = pi.get_grads(
                    states=np.vstack(episode_log[:, 0]),
                    actions=episode_log[:, 1],
                    rewards=discounted_rewards)
                for i, g in enumerate(pe_grads):
                    grad_buffer_pe[i] += g

                if isinstance(V, VApproximationWithNN):
                    ve_grads = V.get_grads(
                        states=np.vstack(episode_log[:, 0]),
                        rewards=discounted_reward_est)
                    for i, g in enumerate(ve_grads):
                        grad_buffer_ve[i] += g

        if ep % batch_size == 0 and ep != 0:
            pi.update(grad_buffer_pe)
            if isinstance(V, VApproximationWithNN):
                V.update(grad_buffer_ve)

            for i, g in enumerate(grad_buffer_pe):
                grad_buffer_pe[i] = g * 0

            if isinstance(V, VApproximationWithNN):
                for i, g in enumerate(grad_buffer_ve):
                    grad_buffer_ve[i] = g * 0

    return total_rewards

    raise NotImplementedError()


def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

