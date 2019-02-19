import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotPredictionVSExpectation(dfs,stateSpace,number):
    data = dfs.iloc[1446*number:1445*(number+1)]

    fig, ax = plt.subplots()

    x = np.linspace(0, 1444, 1445)

    y_predict = data["actions"]
    y_expected = data["activity_formated"]

    ax.plot(x, y_expected, "-", label="Expected")
    ax.plot(x, y_predict, "o", label="Predicted")

    y_missing = []
    for epa in data["elementary_activity"]:
        if epa == "missing":
            y_missing.append(8)
        else:
            y_missing.append(6)

    ax.plot(x, y_missing, "--", label="Smartphone missing (6=no,8=yes)")

    labels = list(stateSpace.values())
    labels = list(map(str, labels))

    ax.legend(shadow=True)
    ax.set_ylabel("State Space")
    ax.set_xlabel("Time (min), starting at wake-up time")
    ax.set_title("Agent's Predictions & Expectations Over Time")

    plt.yticks(list(range(len(labels))), labels)
    plt.show()

def plotBeliefStateOverTime(dfs,stateSpace,number):
    data = dfs.iloc[1446*number:1445*(number+1)]

    fig, ax = plt.subplots()
    ys = []

    x = np.linspace(0, 1444, 1445)
    for i in range(17):
        y = data["bs_" + str(i)]
        ys.append(y)
        ax.plot(x, y, "--", label=str(stateSpace[i]))

    ax.legend(shadow=True)
    ax.set_ylabel("Probability of given State")
    ax.set_xlabel("Time (min), starting at wake-up time")
    ax.set_title("Agent's Belief State Over Time")
    plt.show()


# for Paper
def plotEnergyExpenditureOverTime(df):
    fig = plt.figure(figsize=(3.6, 3.6))
    subplot = fig.add_subplot(1, 1, 1, position=[0.15, 0.15, 0.75, 0.75])
    exp = list(df["expected_met"].iloc[:1445])
    pred = list(df["predicted_met"].iloc[:1445])
    t = np.linspace(0, 1444, 1445)
    plt.plot(t, np.cumsum(exp))
    plt.plot(t, np.cumsum(pred))
    plt.plot((587, 587), (16, 0), "k--", linewidth=1.5)
    subplot.text(1000, 15, 'Expected', color='tab:blue', fontsize=9, weight='medium', horizontalalignment="center",
                 verticalalignment="center")
    subplot.text(250, 20, 'Predicted', color='tab:orange', fontsize=9, weight='medium', horizontalalignment="center",
                 verticalalignment="center")
    subplot.text(800, 0, 't=587', color='k', fontsize=9, weight='medium', horizontalalignment="center",
                 verticalalignment="center")
    subplot.set_ylabel("Cumulative Energy Expenditure (MET)", fontsize=8)
    subplot.set_xlabel("Time (min)", fontsize=8)
    subplot.set_title("Cumulative Energy Expenditure Over Time", fontsize = 9, weight="semibold")
    fig.savefig('energy_expenditure_over_time.eps')

# for paper
def plotPredictionVSExpectationPaper(dfs,stateSpace,number):
    data = dfs.iloc[1446*number:1445*(number+1)]
    data = data.iloc[487:687]

    fig = plt.figure(figsize=(7.6, 3.6))
    subplot = fig.add_subplot(1, 1, 1, position=[0.15, 0.15, 0.75, 0.75])

    # x = np.linspace(0, 1444, 1445)
    x = np.linspace(487,686,200)

    y_predict = data["actions"]
    y_expected = data["activity_formated"]

    subplot.plot(x, y_expected, "-", label="Expected")
    subplot.plot(x, y_predict, ".", label="Predicted",linewidth=0.5)

    plt.axvspan(487, 565, color='r', alpha=0.2, lw=0)
    plt.axvspan(565, 587, color='g', alpha=0.2, lw=0)
    plt.axvspan(587, 686, color='b', alpha=0.2, lw=0)

    labels = list(stateSpace.values())
    labels = list(map(str, labels))

    plt.plot((587, 587), (16, 0), "k--", linewidth=1.5)

    subplot.text(615, 15, 'Expected', color='tab:blue', fontsize=9, weight='medium', horizontalalignment="center",
                 verticalalignment="center")
    subplot.text(615, 14, 'Predicted', color='tab:orange', fontsize=9, weight='medium', horizontalalignment="center",
                 verticalalignment="center")
    subplot.text(597, 0, 't=587', color='k', fontsize=9, weight='medium', horizontalalignment="center",
                 verticalalignment="center")

    # subplot.legend(shadow=True)
    subplot.set_ylabel("State Space", fontsize=8)
    subplot.set_xlabel("Time (min)", fontsize=8)
    subplot.set_title("Agent's Predictions & Expectations Over Time", fontsize=9, weight="semibold")

    plt.yticks(list(range(len(labels))), labels)
    plt.show()
    fig.savefig('prediction_vs_expectation_over_time.png')