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
	

env = CliffWalkingEnv()



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
Expected SARSA Algorithm = 1 step TREE BACKUP
"""

def one_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        state = env.reset()

        #steps within each episode
        for t in itertools.count():
            #pick the first action
            #choose A from S using policy derived from Q (epsilon-greedy)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

            #reward and next state based on the action chosen according to epislon greedy policy
            next_state, reward, done, _ = env.step(action)
            
            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t


            #pick the next action
            # we want an expectation over the next actions 
            #take into account how likely each action is under the current policy

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )


            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            #Update rule in Expected SARSA
            td_target = reward + discount_factor * V
            td_delta = td_target - Q[state][action]

            Q[state][action] += alpha * td_delta


            if done:
                break
            state = next_state

    return Q, stats




"""
Two Step Tree Backup
"""

def two_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

        state = env.reset()

        #steps within each episode
        for t in itertools.count():
            #pick the first action
            #choose A from S using policy derived from Q (epsilon-greedy)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

            #reward and next state based on the action chosen according to epislon greedy policy
            next_state, reward, _ , _ = env.step(action)
            
            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])


            next_next_state, next_reward, done, _ = env.step(next_action)
    
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

            next_V = np.sum(next_next_action_probs * Q[next_next_state])            


            # print "Next Action:", next_action
            # print "Next Action probs :", next_action_probs

            #Main Update Equations for Two Step Tree Backup
            Delta = next_reward + discount_factor * next_V - Q[next_state][next_action]

            # print "Delta :", Delta

            # print "Next Action Prob ", np.max(next_action_probs)

            next_action_selection_probability = np.max(next_action_probs)

            td_target = reward + discount_factor * V +  discount_factor *  next_action_selection_probability * Delta


            td_delta = td_target - Q[state][action]


            Q[state][action] += alpha * td_delta


            if done:
                break

            state = next_state

    return Q, stats


"""
Three Step Tree Backup
"""


def three_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        state = env.reset()

        #steps within each episode
        for t in itertools.count():
            #pick the first action
            #choose A from S using policy derived from Q (epsilon-greedy)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
            next_state, reward, _ , _ = env.step(action)

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )
            next_next_state, next_reward, _, _ = env.step(next_action)
    

            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)
            next_next_next_state, next_next_reward, done, _ = env.step(next_next_action)
 
            next_next_next_action_probs  = policy(next_next_next_state)
            next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

 

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t



            #updates for the Three Step Tree Backup

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            One_Step = reward + discount_factor * V



            next_V = np.sum(next_next_action_probs * Q[next_next_state])            
            Delta_1 = next_reward + discount_factor * next_V - Q[next_state][next_action]
            next_action_selection_probability = np.max(next_action_probs)            

            Two_Step = discount_factor * next_action_selection_probability * Delta_1



            next_next_V = np.sum(next_next_next_action_probs * Q[next_next_next_state])
            Delta_2 = next_next_reward + discount_factor * next_next_V - Q[next_next_state][next_next_action]
            next_next_action_selection_probability = np.max(next_next_action)

            Three_Step = discount_factor * next_action_selection_probability * discount_factor * next_next_action_selection_probability * Delta_2




            td_target = One_Step + Two_Step + Three_Step 

            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta




            if done:
                break

            state = next_state

    return Q, stats





def plot(stats, smoothing_window=200, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(rewards_smoothed, label="Two Step Tree Backup")

    plt.legend(handles=[cum_rwd])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Tree Backup Algorithm")
    plt.show()


    return fig



def multiple_plots(stats1,  stats3,  stats5, smoothing_window=200, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="Expected SARSA (One-Step Tree Backup)")
    # cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Discount Factor = 0.3")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Two Step Tree Backup")
    # cum_rwd_4, = plt.plot(rewards_smoothed_4, label="Discount Factor = 0.7")
    cum_rwd_5, = plt.plot(rewards_smoothed_5, label="Three Step Tree Backup")


    plt.legend(handles=[cum_rwd_1,  cum_rwd_3,  cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Expected SARSA with n-Step Tree Backup on Cliff World MDP")
    plt.show()


    return fig





def main():

    Number_Episodes = 1500

    print "Two Step Tree Backup"
    two_step_tree_backup_Q, stats_two_step_tree_backup_1 = one_step_tree_backup(env, Number_Episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1)
    two_step_tree_backup_Q, stats_two_step_tree_backup_2 = two_step_tree_backup(env, Number_Episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1)
    two_step_tree_backup_Q, stats_two_step_tree_backup_3 = three_step_tree_backup(env, Number_Episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1)

    multiple_plots(stats_two_step_tree_backup_1, stats_two_step_tree_backup_2, stats_two_step_tree_backup_3)


    # print "Trial"
    # Q_sigma, stats_Q_sigma = q_sigma(env, Number_Episodes)
    # trial_plot(stats_Q_sigma)



if __name__ == '__main__':
    main()


