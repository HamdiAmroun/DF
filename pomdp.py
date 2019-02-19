from random import randint
import numpy as np

class RewardManager:
    codeToActivityDict = {
        5035 : 0,
        7025 : 1,
        7030 : 2,
        7040 : 3,
        9045 : 4,
        9055 : 5,
        9070 :6,
        10074 : 7,
        11580 : 8,
        13030 : 9,
        13040 : 10,
        16016 : 11,
        17070 : 12,
        17133 : 13,
        17151 : 14,
        17152 : 15,
        17190 : 16
    }

    def __init__(self,stateSpace,actionSpace):
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace
        self.rewards = self.initRewards(stateSpace,actionSpace)

        self.changeRewards(self.codeToActivityDict[9045],new_reward=50)
        self.changeRewards(self.codeToActivityDict[7040], new_reward=50)
        self.changeRewards(self.codeToActivityDict[7025], new_reward=50)
        self.changeRewards(self.codeToActivityDict[9070], new_reward=50)
        self.changeRewards(self.codeToActivityDict[5035], new_reward=50)

    def __str__(self):
        return str(self.rewards)

    def initRewards(self,stateSpace,actionSpace):
        rewards = {}

        for state in stateSpace.keys():
            rewards[state] = {}
            for action in actionSpace.keys():
                if state == action:
                    rewards[state][action] = 1
                else:
                    rewards[state][action] = -1

        return rewards

    def changeRewards(self,state,action = None,new_reward = 1):
        action = state if action == None else action #if action isn't set, set it to state
        for action_bis in self.actionSpace.keys():
            self.rewards[state][action_bis] = new_reward if action == action_bis else - new_reward

    def getReward(self,belief_states,action):
        # p(b,a) = sum(r(s,a)*b(s))
        reward = 0
        for state in self.stateSpace:
            reward += self.rewards[state][action] * belief_states[state]
        return reward

    def getExactReward(self,state,action):
        # None for the action parameter is used to simulate the wrong action, a wrong prediction
        return self.rewards[state][action] if action is not None else -self.rewards[state][state]

class ValueIteration:
    EPSILON = 0.0001

    def __init__(self,stateSpace,actionSpace):
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace
        self.valueFunction = self.initValueFunction(stateSpace)
        self.policy = self.initPolicy(self.stateSpace)
        self.rewardManager = RewardManager(stateSpace,actionSpace)

    def initValueFunction(self,stateSpace):
        value_function = [0] * len(stateSpace.keys())
        return value_function

    def initPolicy(self,stateSpace):
        policy = [0] * len(stateSpace.keys())
        return policy

    def getBestAction(self, belief_state):
        best_action = None
        best_action_value = None
        # for all the possible actions
        for action in self.actionSpace:
            # compute the value of the action
            value = 0
            for state in self.stateSpace:
                value += belief_state[state] * self.rewardManager.getExactReward(state, action)

            # if it is better than the current best action, make it the best action
            if (best_action_value is None and best_action is None) or value >= best_action_value:
                best_action_value = value
                best_action = action

        return best_action



