def plothelper(filename,):
    with open(filename, "r") as file:
        lines = file.readlines()
        lines_list = [i.split() for i in lines]
        y = [re.sub("[(),]", "", element[1] for element in lines_list]
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