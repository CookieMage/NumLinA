import matplotlib.pyplot as plt
import linear_solvers as linsol
import poisson_problem as pp
from block_matrix import BlockMatrix
import numpy as np

def test_1(max_n, slices_n, max_d, slices_d):
    n_list = np.logspace(2, max_n, slices_n, dtype=int)
    d_list = np.logspace(1, max_d, slices_d, dtype=int)
    
    hat_u = pp.bsp_1
    u = pp.pp_zu_bsp_1
    
    for n in n_list:
        for d in d_list:
            errors += pp.compute_error(d, n, hat_u, u)


def graph_sparse_dense(d=1, x=5, n=25):
    x_values = np.logspace(0.4, x, dtype=int, num=n)
    x_values = [int(x)**d for x in x_values]
    
    data = [[],[],[],[]]
    for x in x_values:
        #print(n)
        # create matrices (d= 1, 2, 3)
        abs_non_zero = (x-1)**d+2*d*(x-2)*(x-1)**(d-1)
        abs_entries = ((x-1)**d)**2


        # get information of sparsity
        data[0] += [abs_non_zero]
        data[1] += [abs_entries]
        data[2] += [x**d]
        data[3] += [(x**d)**2]
    # create the plot
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    #plt.title(f"{num}. Dimension", fontsize=20)
    plt.ylabel("Eintraege", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("N", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    # plot data
    plt.plot(x_values, data[0], label = f"sparse matrix d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[1], label = f"dense matrix  d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[2], label = f"O(n)=n^{d}", linewidth=2, linestyle="dotted")
    plt.plot(x_values, data[3], label = f"O(n)=(n^{d})^2", linewidth=2, linestyle="dotted")
    

    plt.legend(fontsize=20, loc="upper left")
    plt.show()

def main():
    # graph_sparse_dense(d=1)
    # graph_sparse_dense(d=2)
    # graph_sparse_dense(d=3)

    if True: 
        n =  100  #anzahl an intervallen pro dimension
        d =  2     #dimension
        h = 1/n    #Intervalllänge
        values_of_b_vecotor = []  #erstellt Vektor für rechte Seite der gleichung
        for i in range(1,1+(n-1)**d):
            x = pp.inv_idx(i,d,n)       #erzeugt eine liste mit den Disrkretisierungspunkten * n
            x = [j/n for j in x]        #bereitet die Diskretisierungspunkte für das einsetzen in die funktion vor
            values_of_b_vecotor.append(pp.pp_zu_bsp_1(x)*(-h**2))   #setzt die Diskretisierungspunkte in eine funktion f ein (rechte seite)
                                                                    #und rechnet f*((-h)^2) statt A * (-1/h^2)
        print(values_of_b_vecotor)              #debuggen
        print(len(values_of_b_vecotor))         #debuggen

        MatrixA = BlockMatrix(d,n)          #erzeugt die koeffizientenmatrix A zu gegebenen n und d
        print(MatrixA.get_sparse().toarray()) #debuggen
        p,l,u = MatrixA.get_lu()        #zerlegt A in p, l und u
        print("------------------- P: \n" )  #debuggen
        print(p)                                #debuggen
        print("------------------- l: \n" )     #debuggen
        print(l)                                #debuggen
        print("------------------- u: \n" ) #debuggen
        print(u)
        if False:  #(True or False) wahlz zwischen lösung durch pyscy oder selbst programierte
            lösung = linsol.solve_lu(p,l,u,values_of_b_vecotor) #löst das lineare gleichungssystem mit pyscy       
        else:
                lösung = linsol.solve_lu_alt(p,l,u,values_of_b_vecotor) #löst das LGS mit eigener Funktion
        print(lösung ,  " OUR vektor von u")

        values_of_u_vecotor = []    #erstellt vektor für analytisch bestimmte werte der Funktion u an den Diskretisierungspunkte
        for i in range(1,1+(n-1)**d):
            x = pp.inv_idx(i,d,n)   
            x = [j/n for j in x]
            values_of_u_vecotor.append(pp.bsp_1(x))  #Soll werte von u an diskretisierungspunkten (analytisch)
        
        print(values_of_u_vecotor , " SOLL values of u vector")  
        
        Max = 0
        for i in range(0, len(lösung)):         #berechnet Maximalen fehler zwischen unsere Lösung und Soll
            if Max < abs(lösung[i]-values_of_u_vecotor[i]):     
                Max = abs(lösung[i]-values_of_u_vecotor[i])
        print(Max , " maximaler Fehler")       
       # testa = np.dot(p,l)
        #testb = np.dot(testa,u)
        #testc = np.dot(testb,lösung)
        #print(testc,"testc")
        #print(values_of_b_vecotor)
    
            


    
if __name__ == "__main__":
    main()