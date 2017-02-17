import gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random

from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict
from lib.envs.gridworld import GridworldEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
	

env = GridworldEnv()



def make_epsilon_greedy_policy(Q, epsilon, nA):

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn

def chosen_action(Q):
	best_action = np.argmax(Q)
	return best_action


def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn



"""
Q(sigma) algorithm
"""

def q_sigma_on_policy(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.9):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #steps within each episode
        for t in itertools.count():
            #take a step in the environment
            # take action a, observe r and the next state
            next_state, reward, done, _ = env.step(action)

            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t


            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )


            #define sigma to be a random variable between 0 and 1?
            sigma = random.randint(0,1)

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            Sigma_Effect = sigma * Q[next_state][next_action] + (1 - sigma) * V



            td_target = reward + discount_factor * Sigma_Effect



            td_delta = td_target - Q[state][action]



            if done:
                break
            action = next_action
            state = next_state

    return Q, stats



def two_step_q_sigma_on_policy(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.9):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #steps within each episode
        for t in itertools.count():

            sigma = random.randint(0,1)

            #sigma = 0

            next_state, reward, done, _ = env.step(action)


            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )

            V = np.sum(next_action_probs * Q[next_state])
            One_Sigma_Effect = sigma * Q[next_state][next_action] + (1 - sigma) * V
            One_Step = reward + discount_factor * One_Sigma_Effect


            next_action_selection_probability = np.max(next_action_probs)
            Two_Step = - discount_factor * (1 - sigma) * next_action_selection_probability * Q[next_state][next_action]



            next_next_state, next_reward, _, _ = env.step(next_action)
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p=next_next_action_probs )

            V_next = np.sum(next_next_action_probs * Q[next_next_state])
            Three_Sigma_Effect = sigma * Q[next_next_state][next_next_action] + (1 - sigma)* V_next
            Int_Three_Step = next_reward + discount_factor * Three_Sigma_Effect
            Three_Step = discount_factor* (1 - sigma)* next_action_selection_probability * Int_Three_Step



            Fourth_Step = -discount_factor * sigma * Q[next_state][next_action]


            Fifth_Sigma_Effect = sigma * Q[next_next_state][next_next_action] + (1 - sigma) * V_next
            Int_Fifth_Step = discount_factor * Fifth_Sigma_Effect
            Int_Int_Fifth_Step = next_reward + Int_Fifth_Step
            Fifth_Step = discount_factor * sigma * Int_Int_Fifth_Step



            td_target = One_Step + Two_Step + Three_Step + Fourth_Step + Fifth_Step


            td_delta = td_target - Q[state][action]



            if done:
                break
            action = next_action
            state = next_state

    return Q, stats







def plot(stats1,  stats3,  smoothing_window=200, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="One Step Q(sigma)")
    # cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Discount Factor = 0.3")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Two Step Q(sigma)")
    # cum_rwd_4, = plt.plot(rewards_smoothed_4, label="Discount Factor = 0.7")


    plt.legend(handles=[cum_rwd_1,  cum_rwd_3])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("On Policy Q(sigma) - Comparing n-step Q(sigma)")
    plt.show()


    return fig




def main():

    Number_Episodes = 1500

    print "q_sigma_on_policy"
    two_step_tree_backup_Q, stats_two_step_tree_backup_1 = q_sigma_on_policy(env, Number_Episodes, discount_factor=1.0, alpha=0.5, epsilon=0.9)
    # two_step_tree_backup_Q, stats_two_step_tree_backup_2 = q_sigma_on_policy(env, Number_Episodes, discount_factor=0.3, alpha=0.5, epsilon=0.9)
    print "q_sigma_on_policy"
    two_step_tree_backup_Q, stats_two_step_tree_backup_3 = two_step_q_sigma_on_policy(env, Number_Episodes, discount_factor=1.0, alpha=0.5, epsilon=0.9)
    # two_step_tree_backup_Q, stats_two_step_tree_backup_4 = q_sigma_on_policy(env, Number_Episodes, discount_factor=0.7, alpha=0.5, epsilon=0.9)

    plot(stats_two_step_tree_backup_1, stats_two_step_tree_backup_3)


    # print "Trial"
    # Q_sigma, stats_Q_sigma = q_sigma(env, Number_Episodes)
    # trial_plot(stats_Q_sigma)



if __name__ == '__main__':
    main()


