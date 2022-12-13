# MT-T7
import csv
from os import listdir
from os.path import isfile, join

#agent, seed, one_goal, num_hosts, num_honeypots, movement_time, won, steps, unique_success, total_success

def averages():
    path = "results/"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for file in onlyfiles:
        line = ""
        agent = ""
        seed = ""
        one_goal = ""
        with open(path + file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            won = {'0':0, '1':0, '-1':0}
            steps = 0
            i = 0
            for row in spamreader:
                if i == 0:
                    line = row[0] + ", " + row[1] + ", " + row[2] + ", " + row[3] + ", " + row[4] + ", " + row[5] + ", "
                    agent = row[0]
                    seed = row[1]
                    one_goal = row[2]
                won[row[6]] = won[row[6]]+1
                steps = steps+int(row[7])
                i += 1
            line += str(won) + ", " + str(steps/i)
        write_to_file(line, agent, seed, one_goal)

def write_to_file(line, agent, seed, one_goal):
    path = "evaluation/" + str(agent) + "_" + str(seed) + "_" + str(one_goal) + ".csv"
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line.split(","))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    #averages()
    columns = ["agent", "seed", "one_goal", "num_hosts", "num_honeypots", "movement_time", "won", "steps", "unique_success", "total_success"]
    path = "results/"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    df = pd.concat((pd.read_csv(join(path, f), header=0, names=columns) for f in onlyfiles))
    #print(df.to_string()) 
    #print(df.info)
    #print(df.describe())

    #print(df.groupby(["won", "agent"])["steps"].mean())
    #print(df.groupby(["won", "agent"])["won"].count())
    print(df.groupby(["won", "agent"])["steps"].mean())
    print(df.groupby(["won", "agent"])["won"].count())
    
    #df.plot(kind="scatter", x="steps", y="won")
    #df["steps"].plot(kind="hist")
    #plt.show()