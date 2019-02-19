import pandas as pd
import numpy as np

from  pomdp.pomdp import RewardManager

def printSuccessRate(dfs,stateSpace,actionSpace):
    # vals = np.array(dfs.groupby("activity_array").mean()["rewards"])
    activities = dfs.groupby("activity_array").mean()

    rewardsManarger = RewardManager(stateSpace,actionSpace)

    res = []

    for index,row in activities.iterrows():
        activity_code = rewardsManarger.codeToActivityDict[index]
        val = row["rewards"]
        reward_coeff_right = rewardsManarger.getExactReward(activity_code,activity_code)
        reward_coeff_wrong = rewardsManarger.getExactReward(activity_code,None)
        a = np.array([[reward_coeff_right, reward_coeff_wrong], [1, 1]])
        b = np.array([1000 * val, 1000])
        res.append(np.linalg.solve(a, b))

    vals = pd.DataFrame(np.array(dfs.groupby("activity_array").mean()["rewards"]))
    succ = []
    for result in res:
        succ.append(result[0] / 1000)
    vals["success"] = succ
    vals.index = dfs.groupby("activity_array").mean()["rewards"].index

    vals.index = [str(x) for x in vals.index]

    print(vals)

def printConfusionMatrices(dfs,stateSpace):
    confusion_matrices = {}
    activity_groups = dfs.groupby("activity_array")
    for activity in stateSpace.values():
        df = activity_groups.get_group(activity)
        y_actu = df.activity_formated
        y_pred = df.actions
        y_actu = pd.Series(y_actu, name='Actual')
        y_pred = pd.Series(y_pred, name='Predicted')
        confusion_matrices[activity] = pd.crosstab(y_actu, y_pred)
        confusion_matrices[activity].index.names = list(
            map(lambda name: str(stateSpace.get(name, name)), df.index.names))
        confusion_matrices[activity].rename(columns=stateSpace, inplace=True)
        confusion_matrices[activity].rename(index=stateSpace, inplace=True)
        confusion_matrices[activity].loc[str(activity) + "_%"] = confusion_matrices[activity].loc[activity] / sum(
            confusion_matrices[activity].loc[activity])

    for i in range(len(stateSpace.keys())):
        index = list(confusion_matrices.keys())[i]
        confusion_matrices[index].to_csv("./results/confusion_matrix_" + str(index), index_label='Predicted')
        print(confusion_matrices[index])