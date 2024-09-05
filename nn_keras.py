import numpy as np
from algo import ValueFunctionWithApproximation
import tensorflow as tf
from tensorflow import keras


class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method

    def create_model(env):
        n_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        observations_input = keras.layers.Input(obs_shape, name='observations_input')
        action_mask = keras.layers.Input((n_actions,), name='action_mask')
        hidden = keras.layers.Dense(32, activation='relu')(observations_input)
        hidden_2 = keras.layers.Dense(32, activation='relu')(hidden)
        output = keras.layers.Dense(n_actions)(hidden_2)
        filtered_output = keras.layers.multiply([output, action_mask])
        model = keras.models.Model([observations_input, action_mask], filtered_output)
        optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1.0)
        model.compile(optimizer, loss='mean_squared_error')
        return model

    def fit_batch(env, model, target_model, batch):
        observations, actions, rewards, next_observations, dones = batch
        # Predict the Q values of the next states. Passing ones as the action mask.
        next_q_values = predict(env, target_model, next_observations)
        # The Q values of terminal states is 0 by definition.
        next_q_values[dones] = 0.0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
        one_hot_actions = np.array([one_hot_encode(env.action_space.n, action) for action in actions])
        history = model.fit(
            x=[observations, one_hot_actions],
            y=one_hot_actions * q_values[:, None],
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        return history.history['loss'][0]

    def __call__(self,s):
        # TODO: implement this method

        return 0.

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        return None

