from pomdp.environment import Environment
from energy_expenditure.activity import EnergyExpenditureManager
from pomdp.agent import Agent
import pandas
import sys
from tqdm import tqdm
import numpy as np
import results_processing

FILE_NAME = "smoothing_2500_rewards_8" # +1/-1

file = open("logs", "w+")
sys.stdout = file

df1 = pandas.read_pickle("scenarios_data/scenario1.pkl")
df2 = pandas.read_pickle("scenarios_data/scenario2.pkl")
df3 = pandas.read_pickle("scenarios_data/scenario3.pkl")
df4 = pandas.read_pickle("scenarios_data/scenario4.pkl")
df5 = pandas.read_pickle("scenarios_data/scenario5.pkl")
df6 = pandas.read_pickle("scenarios_data/scenario6.pkl")
df7 = pandas.read_pickle("scenarios_data/scenario7.pkl")
df8 = pandas.read_pickle("scenarios_data/scenario8.pkl")
df9 = pandas.read_pickle("scenarios_data/scenario9.pkl")
df10 = pandas.read_pickle("scenarios_data/scenario10.pkl")

dataframes = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]

EEM = EnergyExpenditureManager("activities.txt")

results_array = []

for i,dataframe in tqdm(enumerate(dataframes)):
    print("********** TRAINING #" + str(i) + " **********")
    df = list(dataframes)
    df.pop(i)

    training_dataframes = pandas.concat(df,ignore_index=True)

    testing_dataframe = dataframe

    environment = Environment(training_dataframes)  # ,save=True)

    agent = Agent(environment)

    results = agent.learn(testing_dataframe)

    # pandas.to_pickle(results,"./results/results"+str(i)+".pkl")

    results_array.append(results)

dfs = pandas.concat(results_array)

dfs = EEM.createEnergyExpenditureOverTimeDF(dfs)

pandas.to_pickle(dfs,"./results/results_" + FILE_NAME + ".pkl")

# print success rate
results_processing.success_rate.printSuccessRate(dfs,environment.stateSpace,environment.actionSpace)
results_processing.success_rate.printConfusionMatrices(dfs,environment.stateSpace)
results_processing.plotting.plotBeliefStateOverTime(dfs,environment.stateSpace,0)
results_processing.plotting.plotPredictionVSExpectation(dfs,environment.stateSpace,0)

file.close()
