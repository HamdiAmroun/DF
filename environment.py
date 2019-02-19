import pandas
from tqdm import tqdm
import numpy as np
import matplotlib.mlab as mlab
import math
from tqdm import tqdm


class Environment:
    elementary_activities = {
        'lie': 0,
        'missing': 1,
        'sit': 2,
        'stairsdown': 3,
        'stairsup': 4,
        'stand': 5,
        'run': 6,
        'walk': 7
    }

    stateSpace = {
        0:5035,
        1:7025,
        2:7030,
        3:7040,
        4:9045,
        5:9055,
        6:9070,
        7:10074,
        8:11580,
        9:13030,
        10:13040,
        11:16016,
        12:17070,
        13:17133,
        14:17151,
        15:17152,
        16:17190
    }

    actionSpace = stateSpace

    SMOOTHEN_WIDENESS = 2500

    def __init__(self, dataframe, load=False, save=False):
        print("***** Init environment *****")
        # self.stateSpace = self.computeStateSpace(dataframe['activity_array'])
        # self.actionSpace = self.computeActionSpace(dataframe['activity_array'])

        print(self.stateSpace)
        self.dataframe = dataframe
        self.dataframe_formated = self.formatDataFrame(self.dataframe, self.stateSpace)

        if load:
            self.loadProbas()
        else:
            self.observationsProbas = self.computeObservationsStateProbas(self.dataframe_formated)
            self.transitionsProbas = self.computeStateTransitionProbas(self.dataframe_formated)

        if save:
            self.saveProbas()

    def computeStateSpace(self, activity_code):
        print("*** computeStateSpace ***")
        dict = {}
        i = 0
        for code in activity_code.unique():
            dict[i] = code
            i += 1
        print(dict)
        return dict

    def computeActionSpace(self, activity_code):
        print("*** computeActionSpace ***")
        dict = {}
        i = 0
        for code in activity_code.unique():
            dict[i] = code
            i += 1
        print(dict)
        return dict

    def formatDataFrame(self, dataframe, stateSpace):
        print("*** formatDataFrame ***")
        df = dataframe.copy()

        # format activity
        mapping = {v: int(k) for k, v in stateSpace.items()}
        df.replace({'activity_array': mapping}, inplace=True)

        # format physical activities
        df.replace({'elementary_activity': self.elementary_activities}, inplace=True)

        # format time
        df["time"] = (df["time"] / (60)).apply(math.floor)

        # format speed
        hist, bins = np.histogram(df["speed_array"],bins=4)
        df["speed_array"] = np.searchsorted(bins, df["speed_array"].values)
        self.speed_bins = bins

        return df

    def getCurrentState(self, step):
        print("* getCurrentState: " + str(step) + " *")
        return self.dataframe_formated["activity_array"].iloc[step]

    def getNextObservation(self, step):
        print("* getNextObservation: " + str(step) + " *")
        return self.dataframe_formated[["elementary_activity", "speed_array", "time"]].iloc[step]

    def getProbaElementaryActivity(self, elementary_activity):
        return self.probas_elementary_activities[int(elementary_activity)]

    def getProbaActivity(self, activity):
        return self.probas_activities[activity]

    def getProbaTime(self, time):
        return self.probas_time

    def getProbaObservationsBelief(self,belief_state,obs1,obs2,obs3):
        return self.observationsProbas[belief_state][(obs1,obs2,obs3)]

    def getProbaSpeed(self, speed):
        # value_bin = np.searchsorted(self.cdf, speed)  # get the bin in which the speed falls into
        # value_bin = np.searchsorted(self.speed_midbins, speed)  # get the bin in which the speed falls into
        # return self.probas_speed[value_bin]  # get proba for being in that bin

        bin = np.searchsorted(self.speed_bins,speed) - 1
        return self.probas_speed[bin]

    def getProbaActivityGivenObservation(self, activity_value, elementary_activity, speed, time):
        return math.exp(
            -np.square((int(activity_value) - sum(np.multiply(self.w, [elementary_activity, speed, time])))) / (
            2 * self.variance)) / math.sqrt(self.variance * 2 * math.pi)

    def getProbaObservation(self, activity_value, elementary_activity, speed, time):
        # P(activity = activity_value | observations = [elementary_activity,speed,time]) = p_obs_act
        p_obs_act = self.getProbaActivityGivenObservation(activity_value, elementary_activity, speed, time)
        p_activity = self.getProbaActivity(int(activity_value))
        p_elementary_activity = self.getProbaElementaryActivity(elementary_activity)
        p_speed = self.getProbaSpeed(speed)
        p_time = self.getProbaTime(time)

        # print(str(activity_value) + ": p_obs_act = " + str(p_obs_act))
        # print(str(activity_value) + ": p_activity = " + str(p_activity))
        # print(str(activity_value) + ": p_elementary_activity = " + str(p_elementary_activity))
        # print(str(activity_value) + ": p_speed = " + str(p_speed))
        # print(str(activity_value) + ": p_time = " + str(p_time))

        # p_activity_2 = 0.0

        # for t in range(0, 24 * 2):
        #     for spd in self.speed_bins[1:]:
        #         for k in self.elementary_activities_dict.keys():
        #             epa = self.elementary_activities_dict[k]
        #             p_activity_2 += self.getProbaElementaryActivity(epa) * self.getProbaSpeed(spd) * self.getProbaTime(
        #                 t) * self.getProbaActivityGivenObservation(activity_value, epa, spd, t)
        #
        # print(str(activity_value) + ": p_activity_2 = " + str(p_activity_2))

        # print("~")
        # if activity_value == 5:
        #     print("P(activity = activity_value | observations = [elementary_activity,speed,time])" + ":" + str(p_obs_act))
        #     print(str(activity_value) + ":" + str(p_activity))
        #     print(str(elementary_activity) + ":" + str(p_elementary_activity))
        #     print(str(speed) + ":" + str(p_speed))
        #     print(str(time) + ":" + str(p_time))
        # print("~")

        return p_obs_act * p_elementary_activity * p_speed * p_time / p_activity

    def getProbaTransition(self, state1, state2):
        return self.transitionsProbas[state1][state2]

    def computeStateTransitionProbas(self, dataframe):
        print("*** computeStateTransitionProbas ***")
        activity_data = dataframe["activity_array"]
        # state transition probas takes the form a of double dictionary
        # first index is s, the init state
        # second index is s', the state transitioning into
        dict = {}
        for item in set(activity_data):
            dict[item] = {}
            for item_bis in set(activity_data):
                dict[item][item_bis] = 0

        # count the transitions
        for i in tqdm(range(len(activity_data) - 1)):
            dict[activity_data.iloc[i]][activity_data.iloc[i + 1]] += 1

        # get probas
        for k in dict.keys():
            total = sum(dict[k].values())
            new_values = np.array(list(dict[k].values())) / total
            for i, k_bis in enumerate(dict[k].keys()):
                dict[k][k_bis] = new_values[i]

        return dict

    def _computeObservationsStateProbas(self, dataframe):
        print("*** computeObservationsStateProbas ***")
        # O(o,s') = P(time,elementary_activity,speed | activity)
        # = P(activity | elementary_activity, time, speed) * P(elementary_activity) * P(time) * P(speed) / P(activity)

        # compute P(elementary_activity)
        print("* P(elementary_activity) = ")
        probas_elementary_activities = [0] * len(self.elementary_activities.keys())
        for data in dataframe['elementary_activity']:
            probas_elementary_activities[data] += 1
        length = len(dataframe.index)
        self.probas_elementary_activities = [count / length for count in probas_elementary_activities]

        print(self.probas_elementary_activities)

        # compute P(time)
        print("* P(time) = ")
        self.probas_time = 1 / (24 * 2)# * 60 * 1000)
        print(self.probas_time)

        # compute P(speed) - source: https://stackoverflow.com/questions/17821458/random-number-from-histogram
        print("* P(speed) = ")
        hist, bins = np.histogram(dataframe["speed_array"])
        self.probas_speed = hist / sum(hist)
        print(self.probas_speed)

        # compute P(activity)

        probas_activities = [0] * len(set(dataframe["activity_array"]))
        for data in dataframe['activity_array']:
            probas_activities[data] += 1
        self.probas_activities = [count / length for count in probas_activities]

        print("* P(activity) = ")
        print(self.probas_activities)
        dict = {}
        for activity in self.stateSpace.keys():
            dict[activity] = {}

        for index, row in tqdm(dataframe.iterrows()):
            if (row["elementary_activity"],row["speed_array"],row["time"]) in dict[row["activity_array"]].keys():
                dict[row["activity_array"]][(row["elementary_activity"],row["speed_array"],row["time"])] += 1
            else :
                dict[row["activity_array"]][(row["elementary_activity"],row["speed_array"],row["time"])] = 1

        if self.SMOOTHEN_WIDENESS != 0:
            dict = self.__smootheCounts(dict)

        for activity in dict.keys():
            total = sum(dict[activity].values())
            for k in dict[activity].keys():
                dict[activity][k] /= total

        self.probas_observations = dict

    def computeObservationsStateProbas(self, dataframe):
        print("*** computeObservationsStateProbas ***")

        dict = {}
        for activity in self.stateSpace.keys():
            dict[activity] = {}

        for index, row in tqdm(dataframe.iterrows()):
            if (row["elementary_activity"], row["speed_array"], row["time"]) in dict[row["activity_array"]].keys():
                dict[row["activity_array"]][(row["elementary_activity"], row["speed_array"], row["time"])] += 1
            else:
                dict[row["activity_array"]][(row["elementary_activity"], row["speed_array"], row["time"])] = 1

        dict = self.__smootheCounts(dict, self.SMOOTHEN_WIDENESS)

        for activity in dict.keys():
            total = sum(dict[activity].values())
            for k in dict[activity].keys():
                dict[activity][k] /= total

        return dict

    def __smootheCounts(self, counts_init, smoothness):
        print("* smootheCounts *")

        if smoothness == 0:
            return counts_init

        new_counts = dict(counts_init)
        for activity in tqdm(counts_init.keys()):
            new_counts[activity] = self.__additive_counts(counts_init[activity], smoothness)

        return new_counts

    def __additive_counts(self, counts, range_gaussian=5):
        new_counts = dict(counts)
        # for every observation tuple
        for tuple in counts.keys():
            (epa, spd, t) = tuple
            # for every time neighbor of the observation tuple
            for i in range(-int(math.sqrt(range_gaussian)), int(math.sqrt(range_gaussian)) + 1):
                # add the to the count a value taken on the gaussian function
                if (epa, spd, t + i) in new_counts.keys():
                    new_counts[(epa, spd, t + i)] += self.__gaussian(i, range_gaussian)
                else:
                    new_counts[(epa, spd, t + i)] = self.__gaussian(i, range_gaussian)
        return new_counts

    def __gaussian(self, x, variance=10):
        mu = 0
        sigma = math.sqrt(variance)
        return mlab.normpdf(x, mu, sigma) / mlab.normpdf(0, mu, sigma)



    def saveProbas(self):
        file = "./tmp/"
        np.save(file + "probas_time", self.probas_time)
        np.save(file + "probas_speed", self.probas_speed)
        np.save(file + "probas_elementary_activities", self.probas_elementary_activities)
        np.save(file + "probas_activities", self.probas_activities)
        np.save(file + "probas_transitions", self.transitionsProbas)
        np.save(file + "speed_bins",self.speed_bins)
        np.save(file + "probas_observations",self.probas_observations)

    def loadProbas(self):
        self.probas_time = np.load("probas_time.npy")
        self.probas_speed = np.load("probas_speed.npy")
        self.probas_elementary_activities = np.load("probas_elementary_activities.npy")
        self.probas_activities = np.load("probas_activities.npy")
        self.transitionsProbas = np.load("probas_transitions.npy").item()
        self.speed_bins = np.load("speed_bins.npy")
        self.probas_observations = np.load("probas_observations.npy").item()

        print("Probas acitivities:")
        print(self.probas_activities)
        print("Probas epa:")
        print(self.probas_elementary_activities)
        print("Probas spd:")
        print(self.probas_speed)
        print("Probas t:")
        print(self.probas_time)
        print("Probas transitions:")
        print(self.transitionsProbas)
        print("Probas observations:")
        print(self.probas_observations)
