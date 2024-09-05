import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """

        delta = alpha * (G - self(s_tau))

        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function

    # initial starting state
    # state = START_STATE

    for episode in range(num_episode):

        state = env.reset()

        # arrays to store states and rewards for an episode
        # space isn't a major consideration, so I didn't use the mod trick
        states = [state]
        rewards = [0]


        # track the time
        time = 0

        # the length of this episode
        T = float('inf')
        while True:
            # go to next time step

            if time < T:
                # choose an action randomly
                # action = get_action()

                action = pi.action(state)

                # next_state, reward = step(state, action)

                next_state, reward, done, info = env.step(action)

                # store new state and new reward
                states.append(next_state)
                rewards.append(reward)


                if done:
                    T = time + 1

            # get the time of the state to update
            update_time = time - n + 1
            if update_time >= 0:
                returns = 0.0
                # calculate corresponding rewards
                for t in range(update_time + 1, min(T, update_time + n) + 1):
                    returns += (gamma ** (t - update_time - 1)) * rewards[t]
                # add state value to the return
                if update_time + n <= T:
                    returns += (gamma ** n) * V(states[update_time + n])
                state_to_update = states[update_time]
                V.update(alpha, returns, state_to_update)
                # state_to_update = states[update_time]
                # # update the value function
                # if not state_to_update in END_STATES:
                #     delta = alpha * (returns - V(state_to_update))
                #     V.update(alpha, returns, state_to_update)
                # w[update_time] += w[update_time] + (alpha * (returns - V(states[update_time])))
            if update_time == T - 1:
                break
            state = next_state
            time += 1


