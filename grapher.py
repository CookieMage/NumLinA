import manage_list as ml
import time
import matplotlib
import matplotlib.pyplot as plt
#plt.style.use('seaborn-v0_8-darkgrid')
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D 
import math
import re

def best_case():
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("best case", fontsize=40)
    plt.ylabel("Zeit in ms", fontsize = 20)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    #ax1.set_xscale('log')
    ax1.grid()


    lines = []
    with open("save_best_case.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_best = [0]*15
    for e in lines_list:
        ax1.scatter(float(e[0]), float(e[1]) * 1000, c="b")
    for j in range(15):
        for i in range(10):
            average_best[j] += float(lines_list[i+(10*j)][1]) * 1000
    for i in range(15):
        average_best[i] = average_best[i]/10
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]

    legend = [Line2D([0], [0], marker='o', color='w', label='Ergebnisse des Experiments',
                              markerfacecolor='b', markersize=10),
              Line2D([0], [0], color='g', label='Durchschnitt')]

    ax1.plot(x, average_best, "g", label="best case", linewidth = 2)
    ax1.legend(handles=legend, fontsize = 20, loc='upper left')

    plt.show()

#
#
#


def verortung():
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Verortung der Laufzeitkomplexität", fontsize=40)
    plt.ylabel("Zeit in ms", fontsize = 20)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    #ax1.set_xscale('log')
    ax1.grid()
    ax1.set_ylim([-1, 10**6])
    ax1.set_xlim([0, 10**7])


    x = [int(10**i) for i in [i*0.5 for i in range(15)]]
    ax1.plot(x, average_best, "g", label="best case", linewidth = 2)

    lines = []
    with open("save_random_case.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_random = [0]*15
    for j in range(15):
        for i in range(10):
            average_random[j] += float(lines_list[i+(10*j)][1]) * 1000
    for i in range(15):
        average_random[i] = average_random[i]/10
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]
    ax1.plot(x, average_random, "y", label="average case", linewidth = 2)


    lines = []
    with open("save_worst_case.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_worst = [0]*15
    for j in range(15):
        for i in range(10):
            average_worst[j] += float(lines_list[i+(10*j)][1]) * 1000
    for i in range(15):
        average_worst[i] = average_worst[i]/10
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]
    ax1.plot(x, average_worst, "r", label="worst case", linewidth = 2)

    genauigkeit = 8

    # Create data
    X=[int(10**(i/genauigkeit)) for i in range(genauigkeit**2-genauigkeit+1)]
    y1=[np.log10(x) for x in X]         #O=log(n)
    y2=[x for x in X]                   #O=n
    y3=[x*np.log10(x) for x in X]       #O=n*log(n)
    y4=[x**2 for x in X]                #O=n**2

    # Basic stacked area chart.
    plt.stackplot(X, y1, y2, y3, y4, labels=['O(n) = n', 'O(n) = n*log(n)', 'O(n) = n^2', 'O(n) = 2^n'], colors=['b', 'g', 'y', 'r'], alpha=[0.1]*4)
    plt.legend(loc='upper left', fontsize = 20)

    plt.show()


#
#
#

def vergleich():
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Vergleich Tim- und Merge-Sort", fontsize=40)
    plt.ylabel("Zeit in ms", fontsize = 20)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    #ax1.set_xscale('log')
    ax1.set_xlim([0, 10**7])
    ax1.grid()

    lines = []
    with open("save_merge.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_merge = [0]*15
    for j in range(15):
        for i in range(10):
            average_merge[j] += float(lines_list[i+(10*j)][1]) * 1000
    for i in range(15):
        average_merge[i] = average_merge[i]/10
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]

    lines = []
    with open("save_merge_best.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_merge_best = [0]*15
    for j in range(15):
        for i in range(5):
            average_merge_best[j] += float(lines_list[i+(5*j)][1]) * 1000
    for i in range(15):
        average_merge_best[i] = average_merge_best[i]/5
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]

    lines = []
    with open("save_merge_worst.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_merge_worst = [0]*15
    for j in range(15):
        for i in range(5):
            average_merge_worst[j] += float(lines_list[i+(5*j)][1])
    for i in range(15):
        average_merge_worst[i] = average_merge_worst[i]/5
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]

    ax1.plot(x, average_best, "g", label="best case Merge-Sort", linewidth = 2)
    ax1.plot(x, average_merge, "y", label="average case Merge-Sort", linewidth = 2)
    ax1.plot(x, average_worst, "r", label="worst case Merge-Sort", linewidth = 2)


    x = [int(10**i) for i in [i*0.5 for i in range(15)]]
    ax1.plot(x, average_best, "g", label="best case Tim-Sort", linestyle='dashed', linewidth = 4)
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]
    ax1.plot(x, average_random, "y", label="average case Tim-Sort", linestyle='dashed', linewidth = 4)
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]
    ax1.plot(x, average_worst, "r", label="worst case Tim-Sort", linestyle='dashed', linewidth = 4)

    plt.legend(fontsize = 20)

    plt.show()

def experimente():
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Experimente", fontsize=40)
    plt.ylabel("Anzahl der Rechenoperationen", fontsize = 20)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    #ax1.set_xscale('log')
    ax1.grid()


    lines = []
    with open("save_#tim_average.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_tim_num = [0]*15
    for e in lines_list:
        ax1.scatter(float(e[0]), float(e[1]) * 1000, c="b")
    for j in range(15):
        for i in range(5):
            average_tim_num[j] += float(lines_list[i+(5*j)][1]) * 1000
    for i in range(15):
        average_tim_num[i] = average_tim_num[i]/5
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]

    legend = [Line2D([0], [0], marker='o', color='w', label='Ergebnisse des Experiments',
                              markerfacecolor='b', markersize=10),
              Line2D([0], [0], color='g', label='Durchschnitt')]

    ax1.plot(x, average_tim_num, "g", label="best case", linewidth = 2)
    ax1.legend(handles=legend, fontsize = 20, loc='upper left')

    plt.show()

def verortung_2():
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Verortung Laufzeitkomplexität", fontsize=40)
    plt.ylabel("Anzahl der Rechenoperationen", fontsize = 20)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    #ax1.set_xscale('log')
    ax1.grid()


    lines = []
    with open("save_#tim_average.txt", "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    average_tim_num = [0]*15
    for e in lines_list:
        ax1.scatter(float(e[0]), float(e[1]) * 1000, c="b")
    for j in range(15):
        for i in range(5):
            average_tim_num[j] += float(lines_list[i+(5*j)][1]) * 1000
    for i in range(15):
        average_tim_num[i] = average_tim_num[i]/5
    x = [int(10**i) for i in [i*0.5 for i in range(15)]]

    legend = [Line2D([0], [0], color='r', label='O(n) = n*log(n)'),
              Line2D([0], [0], color='g', label='Timsort')]
              #Line2D([0], [0], color='y', label='O(n) = n')]

    ax1.plot(x, average_tim_num, "g", label="Timsort", linewidth = 2)


    x2 = [1.1] + x
    x2.pop(1)

    y3=[i*np.log10(i) for i in x2]       #O=n*log(n)
    y3 = [i*30000 for i in y3]

    plt.plot(x2, y3, "r", label="O = n*log(n)", linewidth = 2)

    #y2=[i*30000 for i in x]                   #O=n
    #
    #plt.plot(x, y2, "y", linewidth = 2)

    ax1.legend(handles=legend, fontsize = 20, loc='upper left')



    plt.show()

def plothelper(filename, comp_time, leng):
    with open(filename, "r") as file:
        lines = file.readlines()
        lines_list = [i.split() for i in lines]
        y = [re.sub("[(),]", "", element[comp_time]) for element in lines_list]
        list_len = len(y)
        try:
            for i,e in enumerate(y):
                y[i] = int(e)
        except Exception:
            for i,e in enumerate(y):
                y[i] = float(e)
    for i in range(list_len//leng):
        for _ in range(leng-1):
            y[i] += y.pop(i+1)
        y[i] /= leng
    y = y[:list_len]
    x = [int(10**i) for i in [i*0.5 for i in range(len(y))]]
    return x, y

def plot(len, title, comp_time, filename_1, name_1, filename_2 = None, name_2 = None, filename_3 = None, name_3 = None, filename_4 = None, name_4 = None):
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=40)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    ax1.grid()
    if comp_time != "time" and comp_time != "comp":
        print("comp_time must be time or comp. not", comp_time)
        exit()
    if comp_time == "comp":
        plt.ylabel("Anzahl der Rechenoperationen", fontsize = 20)
        comp_time = 1
    else:
        plt.ylabel("Zeit in ms", fontsize = 20)
        comp_time = 2
    
    y1 = []
    x, y1 = plothelper(filename_1, comp_time, len)
    plt.plot(x, y1, "b", label = name_1, linewidth = 2, linestyle="solid")
    
    if filename_2 != None:
        y2 = []
        x, y2 = plothelper(filename_2, comp_time, len)
        plt.plot(x, y2, "g", label = name_2, linewidth = 2, linestyle="dashed")
        
    if filename_3 != None:
        y3 = []
        x, y3 = plothelper(filename_3, comp_time, len)
        plt.plot(x, y3, "r", label = name_3, linewidth = 2, linestyle="dotted")
        
    if filename_4 != None:
        y4 = []
        x, y4 = plothelper(filename_4, comp_time, len)
        plt.plot(x, y4, "y", label = name_4, linewidth = 2, linestyle="dashdot")
        
    plt.legend(fontsize=20)
    plt.show()


plot(5, "worst case", "time", "worst_tim.txt", "Timsort", "worst_rec.txt", "recursive Mergesort", "worst_loop.txt", "iterative Mergesort")
plot(5, "worst case", "comp", "worst_tim.txt", "Timsort", "worst_rec.txt", "recursive Mergesort", "worst_loop.txt", "iterative Mergesort")
plot(5, "best case", "time", "best_tim.txt", "Timsort", "best_rec.txt", "recursive Mergesort", "best_loop.txt", "iterative Mergesort")
plot(5, "best case", "comp", "best_tim.txt", "Timsort", "best_rec.txt", "recursive Mergesort", "best_loop.txt", "iterative Mergesort")
plot(5, "average case", "time", "average_tim.txt", "Timsort", "average_rec.txt", "recursive Mergesort", "average_loop.txt", "iterative Mergesort")
plot(5, "average case", "comp", "average_tim.txt", "Timsort", "average_rec.txt", "recursive Mergesort", "average_loop.txt", "iterative Mergesort")
plot(5, "Vergleich zu Insertionsort", "time", "insertion_tim.txt", "Timsort", "insertion.txt", "Insertionsort")
plot(5, "Vergleich zu Insertionsort", "comp", "insertion_tim.txt", "Timsort", "insertion.txt", "Insertionsort")




fig1, ax1 = plt.subplots(figsize=(5, 5))
plt.yscale("log")
plt.xscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Laufzeitkomplexität", fontsize=40)
plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
plt.ylabel("Zeit ins ms", fontsize = 20)
ax1.grid()
with open("average_rec.txt", "r") as file:
    lines = file.readlines()
    lines_list = [i.split() for i in lines]
    y = [re.sub("[(),]", "", element[1]) for element in lines_list]
    list_len = len(y)
    try:
        for i,e in enumerate(y):
            y[i] = int(e)
    except Exception:
        for i,e in enumerate(y):
            y[i] = float(e)
for i in range(list_len//5):
    for _ in range(4):
        y[i] += y.pop(i+1)
    y[i] /= 5
y_tim = y[:list_len]
x_tim = [int(10**i) for i in [i*0.5 for i in range(len(y))]]

#with open("insertion.txt", "r") as file:
#    lines = file.readlines()
#    lines_list = [i.split() for i in lines]
#    y = [re.sub("[(),]", "", element[1]) for element in lines_list]
#    list_len = len(y)
#    try:
#        for i,e in enumerate(y):
#            y[i] = int(e)
#    except Exception:
#        for i,e in enumerate(y):
#            y[i] = float(e)
#for i in range(list_len//5):
#    for _ in range(4):
#        y[i] += y.pop(i+1)
#    y[i] /= 5
#y_ins = y[:list_len]
#x_ins = [int(10**i) for i in [i*0.5 for i in range(len(y))]]

#with open("nlogn.txt", "r") as file:
#    lines = file.readlines()
#    lines_list = [i.split() for i in lines]
#    y = [re.sub("[(),]", "", element[1]) for element in lines_list]
#    list_len = len(y)
#    try:
#        for i,e in enumerate(y):
#            y[i] = int(e)
#    except Exception:
#        for i,e in enumerate(y):
#            y[i] = float(e)
#y_log = y[:list_len]
#x_log = [int(10**i) for i in range(len(y))]

with open("squared.txt", "r") as file:
    lines = file.readlines()
    lines_list = [i.split() for i in lines]
    y = [re.sub("[(),]", "", element[1]) for element in lines_list]
    list_len = len(y)
    try:
        for i,e in enumerate(y):
            y[i] = int(e)
    except Exception:
        for i,e in enumerate(y):
            y[i] = float(e)
y_squared = y[:list_len]
x_squared = [int(10**i) for i in range(len(y))]

#plt.plot(x_ins, y_ins, "b", label = "Insertionsort", linewidth = 2, linestyle = "solid")
plt.plot(x_tim, y_tim, "b", label = "recursive Mergesort", linewidth = 2, linestyle = "dashed")
#plt.plot(x_log, y_log, "g", label = "O(n)=n*lg(n)", linewidth = 2, linestyle = "dotted")
plt.plot(x_squared, y_squared, "r", label = "O(n)=n^2", linewidth = 2, linestyle = "dashdot")


plt.legend(fontsize=20)
plt.show()