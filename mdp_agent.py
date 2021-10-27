# !/usr/bin/env python3
# File:     08_sdp: mdp_agent.py
# Author:   Lukas Malek
# Date:     05.04.2021
# Course:   KUI 2021

# SUMMARY OF THE FILE
# Value and policy iteration to find the right policy for each state in the
# defined maze (the rewards and probabilities are set, reinforcement learning
# where the probabilities are unknown will be in the following assignment

# SOURCE
# Main source of inspiration for this assignemnt was from these materials
# https://cw.fel.cvut.cz/wiki/_media/courses/be5b33kui/lectures/06_mdp.pdf
# https://cw.fel.cvut.cz/wiki/_media/courses/b3b33kui/prednasky/07_mdp.pdf


import kuimaze
import random
import os
import time
import sys
import copy

# VALUE INITIALIZATION
PROBS = [0.8, 0.1, 0.1, 0]
EPSILON = 0.03
GRAD = (0, 0)
SKIP = False
# SAVE_EPS = False
# VERBOSITY = 0

# BASIC GRIDS FOR DEBUGING
GRID_WORLD4 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
               [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 0, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

REWARD_NORMAL_STATE = -0.04
REWARD_GOAL_STATE = 1
REWARD_DANGEROUS_STATE = -1

GRID_WORLD3_REWARDS = [[REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_GOAL_STATE],
                       [REWARD_NORMAL_STATE, 0, REWARD_NORMAL_STATE,
                           REWARD_DANGEROUS_STATE],
                       [REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE]]


def wait_n_or_s():
    def wait_key():
        '''
        returns key pressed ... works only in terminal! NOT in IDE!
        '''
        result = None
        if os.name == 'nt':
            import msvcrt
            result = msvcrt.getch()
        else:
            import termios
            fd = sys.stdin.fileno()

            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            try:
                result = sys.stdin.read(1)
            except IOError:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        return result

    '''
    press n - next, s - skip to end ... write into terminal
    '''
    global SKIP
    x = SKIP
    while not x:
        key = wait_key()
        x = key == 'n'
        if key == 's':
            SKIP = True
            break


def get_visualisation_values(dictvalues):
    if dictvalues is None:
        return None
    ret = []
    for key, value in dictvalues.items():
        ret.append({'x': key[0], 'y': key[1], 'value': value})
    return ret


def init_policy(problem):
    '''
    Initialize all policies except the goal state to random action
    :param problem: problem - object, for us it will be kuimaze.Maze object
    :return: dictionary of policies, indexed by state coordinates
    '''
    policy = dict()
    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            policy[state.x, state.y] = None
            continue
        actions = [action for action in problem.get_actions(state)]
        policy[state.x, state.y] = random.choice(actions)
    return policy


def init_utils(problem):
    '''
    Initialize all state utilities to zero except the goal states
    :param problem: problem - object, for us it will be kuimaze.Maze object
    :return: dictionary of utilities, indexed by state coordinates
    '''
    utils = dict()
    x_dims = problem.observation_space.spaces[0].n
    y_dims = problem.observation_space.spaces[1].n

    for x in range(x_dims):
        for y in range(y_dims):
            utils[(x, y)] = 0

    for state in problem.get_all_states():
        problem.get_state_reward(state)
        utils[(state.x, state.y)] = state.reward
    return utils


def policy_evaluation(state, action, problem, utils):
    '''
    Return the evaluation of the specified action from the current state.
    Because of the uncertainty the probabilities of each direction are
    defined in variable PROBS = [0.8, 0.1, 0.1, 0]
    '''
    probabilities = problem.get_next_states_and_probs(
        state, action)
    temp_val = 0
    for probability in probabilities:
        temp_val += utils[probability[0]] * probability[1]
    return temp_val


def max_of_all_actions(state, problem, utils):
    '''
    Return the maximum value we can get from all possible actions from the
    current state
    '''
    maximum_of_all_states = float('-inf')
    for action in problem.get_actions(state):
        temp_val = policy_evaluation(state, action, problem, utils)
        if temp_val > maximum_of_all_states:
            maximum_of_all_states = temp_val
    return maximum_of_all_states


def best_action_from_current_state(state, problem, utils):
    '''
    Return the action how to reach the neighbor with highest value from the 
    current state s.
    '''
    max_neighbour_value = float('-inf')
    for action in problem.get_actions(state):
        temp_val = policy_evaluation(state, action, problem, utils)
        if temp_val > max_neighbour_value:
            max_neighbour_value = temp_val
            policy = action
    return policy


def find_policy_via_policy_iteration(problem, discount_factor):
    '''
    Value evaluation using the algorithm at the following presentation
    https://cw.fel.cvut.cz/wiki/_media/courses/be5b33kui/lectures/07_mdp.pdf
    on the slide number 24
    '''
    policy = init_policy(problem)
    utils = init_utils(problem)  # utils = value of each state S
    changed = True
    while changed:
        changed = False
        utils_ = copy.deepcopy(utils)
        for state in problem.get_all_states():
            if problem.is_terminal_state(state):
                continue
            utils[state.x, state.y] = state.reward + \
                discount_factor*max_of_all_actions(
                state, problem, utils_)
            if max_of_all_actions(state, problem, utils) > max_of_all_actions(
                    state, problem, utils_):  # there it is slightly wrong, we
                # souhld not use the second max_of_all_actions - for more info
                # check the equation from ctu lectures
                policy[state.x, state.y] = best_action_from_current_state(
                    state, problem, utils)
                changed = True
    return(policy)


def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    '''
    Value evaluation using the algorithm at the following presentation
    https://cw.fel.cvut.cz/wiki/_media/courses/be5b33kui/lectures/06_mdp.pdf
    on the slide number 24
    '''
    policy = init_policy(problem)
    utils_ = init_utils(problem)
    while True:
        utils = copy.deepcopy(utils_)
        difference = 0
        for state in problem.get_all_states():
            if problem.is_terminal_state(state):
                continue
            utils_[(state.x, state.y)] = state.reward + \
                discount_factor*max_of_all_actions(
                state, problem, utils)
            if utils_[(state.x, state.y)] - utils[(state.x, state.y)] > difference:
                difference = utils_[(state.x, state.y)] - \
                    utils[(state.x, state.y)]
        if difference < epsilon * (1 - discount_factor) / discount_factor:
            break
    for state in problem.get_all_states():
        if problem.is_terminal_state(state):
            policy[state.x, state.y] = None
            continue
        policy[(state.x, state.y)] = best_action_from_current_state(
            state, problem, utils)
    return(policy)


if __name__ == "__main__":
    MAP = 'maps/normal/normal1.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS,
                          grad=GRAD, node_rewards=GRID_WORLD3_REWARDS)
    # env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=None)
    # env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
    env.reset()

    print('====================')
    print('works only in terminal! NOT in IDE!')
    print('press n - next')
    print('press s - skip to end')
    print('====================')

    # VISUALISE THE MAZE WITH REWARD
    utils = init_utils(env)
    env.visualise(get_visualisation_values(utils))
    env.render()
    wait_n_or_s()

    # VISUALISE THE VALUE ITERATION
    value = find_policy_via_value_iteration(env, 0.9999, EPSILON)
    env.visualise(get_visualisation_values(value))
    env.render()
    wait_n_or_s()

    # VISUALISE THE POLICY ITERATION
    policy = find_policy_via_policy_iteration(env, 0.9999)
    env.visualise(get_visualisation_values(policy))
    env.render()
    wait_n_or_s()
