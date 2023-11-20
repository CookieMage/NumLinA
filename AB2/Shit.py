
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


def inv_idx(m, d, n):
    """ Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.
    
    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    M = m-1
    nx = [1] * d 
    print(nx,"Creat")
    print(len(nx),"Listen Länge")
    for i in range(len(nx),0,-1):
        print(i,"das ist i")
        nx[i-1] = nx[i-1] + (M // ((n-1)**(i-1)))
        print(nx,"das ist nx zu",i)
        M = M % (n-1)**(i-1)

    return(nx)

print(inv_idx(9,3,4))