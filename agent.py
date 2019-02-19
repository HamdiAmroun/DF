from pomdp.pomdp import ValueIteration
import numpy as np
import sys
from tqdm import tqdm


class Agent:
    step = 0

    def __init__(self, environment):
        self.environment = environment
        self.algorithm = ValueIteration(environment.stateSpace, environment.actionSpace)
        self.rewards = 0
        self.expected_average_rewards = 0

    def initBeliefStates(self):
        return [1 / len(self.environment.stateSpace)] * len(self.environment.stateSpace)

    def learn(self, testing_environement):
        # format the testing environement
        # WARNING - speed formating may be inacurate
        testing_environement_formated = self.environment.formatDataFrame(testing_environement,
                                                                         self.environment.stateSpace)

        print("***** Learn *****")
        belief_states = self.initBeliefStates()
        rewards_list = []
        actions = []
        bs_list = []

        # for every step in the testing dataset
        for index, data in tqdm(testing_environement_formated[
                                    ["activity_array", "elementary_activity", "speed_array", "time"]].iterrows()):

            # get the current activity
            activity = data["activity_array"]
            # get the current observations
            observations = data[["elementary_activity", "speed_array", "time"]]
            # update the belief state given the new observations
            belief_states = self.updateBeliefStates(observations, belief_states)
            bs_list.append(belief_states)
            # choose an action according to the new belief state
            action = self.algorithm.getBestAction(belief_states)
            actions.append(action)
            # get rewards
            rewards = self.algorithm.rewardManager.getExactReward(activity, action)
            rewards_list.append(rewards)

            print("*** Step #" + str(index) + ": predicted=" + str(action) + ", true=" + str(
                activity) + ", rewards=" + str(sum(rewards_list)) + " ***")
            print("observations: ")
            print(observations)
            print("belief states:")
            print(belief_states)

        testing_environement_results = testing_environement.copy()
        testing_environement_results["activity_formated"] = testing_environement_formated["activity_array"]
        testing_environement_results["time_formated"] = testing_environement_formated["time"]
        testing_environement_results["speed_formated"] = testing_environement_formated["speed_array"]
        testing_environement_results["epa_formated"] = testing_environement_formated["elementary_activity"]
        bs_list = np.array(bs_list).transpose()
        for i in range(len(self.environment.stateSpace)):
            testing_environement_results["bs_" + str(i)] = bs_list[i]
        testing_environement_results["actions"] = actions
        testing_environement_results["rewards"] = rewards_list

        return testing_environement_results

    def updateBeliefStates(self, observations, belief_states):
        # normalization factor = sum(sum( O(o|s'')p(s''|s)b(s)
        denom = 0
        for belief_state in self.environment.stateSpace:
            if (observations[0], observations[1], observations[2]) in self.environment.observationsProbas[
                belief_state].keys():
                observations_proba = self.environment.getProbaObservationsBelief(belief_state,observations[0], observations[1], observations[2])
            else:
                observations_proba = 0.0
            for belief_state_bis in self.environment.stateSpace:
                denom += observations_proba * self.environment.getProbaTransition(belief_state_bis, belief_state) * \
                         belief_states[belief_state_bis]

        # new belief states = b(s')
        new_belief_states = [0] * len(self.environment.stateSpace)
        for belief_state in self.environment.stateSpace:
            # num = O(o|s') sum(p(s'|s)b(s))
            if (observations[0], observations[1], observations[2]) in self.environment.observationsProbas[
                belief_state].keys():
                observations_proba = self.environment.getProbaObservationsBelief(belief_state,observations[0], observations[1], observations[2])
            else:
                observations_proba = 0.0
            num = 0
            for belief_state_bis in self.environment.stateSpace:
                num += self.environment.getProbaTransition(belief_state_bis, belief_state) * belief_states[
                    belief_state_bis]
            num *= observations_proba

            new_belief_states[belief_state] = num / denom if denom != 0 else self.initBeliefStates()[belief_state]

        return new_belief_states

    def takeAction(self, action, belief_states):
        return self.algorithm.rewardManager.getReward(belief_states, action)
