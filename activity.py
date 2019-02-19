class EnergyExpenditureManager:
    activityToCodeDict = {
        0: 5035,
        1: 7025,
        2: 7030,
        3: 7040,
        4: 9045,
        5: 9055,
        6: 9070,
        7: 10074,
        8: 11580,
        9: 13030,
        10: 13040,
        11: 16016,
        12: 17070,
        13: 17133,
        14: 17151,
        15: 17152,
        16: 17190
    }

    def __init__(self,file):
        self.activity_array = self.loadActivities(file)

    def loadActivities(self,file):
        activity_array = []

        with open(file,"r") as f:
            lines = f.read().splitlines()

        for line in lines:
            tab = line.split(' | ')
            code = int(tab[0])
            met = float(tab[1].replace(",","."))
            label = tab[2]
            activity = Activity(code,met,label)
            activity_array.append(activity)

        return activity_array

    def getActivityFromCode(self,code):
        for activity in self.activity_array:
            if activity.code == code:
                return activity

        raise NameError("Can't find an activity for the given code: " + str(code))

    def printActivities(self):
        for activity in self.activity_array:
            print(activity)

    def createEnergyExpenditureOverTimeDF(self,results):
        exp = []
        pred = []
        for index,row in results.iterrows():
            activity_exp = self.getActivityFromCode(row["activity_array"])
            activity_pred = self.getActivityFromCode(self.activityToCodeDict[row["actions"]])
            exp.append(activity_exp.met / 60)
            pred.append(activity_pred.met / 60)

        results["expected_met"] = exp
        results["predicted_met"] = pred

        return results

class Activity:
    def __init__(self,code,met,label):
        self.code = code
        self.label = label
        self.met = met

    def __str__(self):
        return "Activity = {'code': " + str(self.code) + ", 'met': " + str(self.met) + ", 'label': " + self.label + "}"