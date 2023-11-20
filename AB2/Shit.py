def idxx(nx, n):
    """ Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """

    num=0 
    num = nx[0]
    if len(nx) > 1:
        num = num + (n-1) * (nx[1]-1)
    if len(nx) > 2:
       num = num + (n-1) * (nx[2]-1)
    if len(nx) > 3:
       num = num + (n-1) * (nx[3]-1) 

    return num 

def idx(nx,n):
    if nx[0] > (n-1):
        raise ValueError
    num = nx[0]
    print(len(nx))
    for i in range(1,len(nx)):
        if nx[i] > (n-1):
            raise ValueError
        num = num + (n-1)**i * (nx[i]-1)
        print(i)
        print("^")

        
    return num
print(idx([10,10,10],11) , "<--")

