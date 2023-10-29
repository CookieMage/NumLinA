'''This is a programm for managing txt-files

Functions
---------
save()
    saves data in a specific format
clear()
    clears a file
main():
    example implementation for using the above defined code
'''

import os

def save(func_name : str, filename : str, data : str, data_1="", data_2=""):
    '''function that appends data in a specific format to a specified file
    
    Parameters
    ----------
    func_name: str
        name of thefunction which produced the data
        file-prefix of file in which the data is saved
    filename : str
        file-suffix of file in which the data is saved
    data : int or list or str
        data that is saved
    data_1 : int or list or str, optional
        data that is saved
    data_2 : int or list or str, optional
        data that is saved
    
    Exceptions
    ----------
    TypeError
        filename is not a str
    '''
    try:
        cwd = os.getcwd()
        if not os.path.exists(cwd + "/experiments"):
            os.makedirs(cwd + "/experiments")
        # haenge die Parameter im Format "iteration data\n" hinten an das Dokument filename an
        with open("experiments/" + func_name + filename, "a", encoding="utf8") as file:
            file.write(str(data)+" "+str(data_1)+" "+str(data_2)+"\n")
    except TypeError as exc:
        raise exc


def clear(func_name, filename):
    '''function for clearing files; if the file did not exist an empty one will be created
    
    Parameters
    ----------
    func_name: str
        name of the function which produced the data that is to be deleted
        file-prefix of file in which the data is be deleted
    filename : str
        file-suffix of file in which the data is to be deleted
    
    Exceptions
    ----------
    TypeError
        filename is not a str
        func_name is not a str
    '''
    try:
        cwd = os.getcwd()
        if not os.path.exists(cwd + "/experiments"):
            os.makedirs(cwd + "/experiments")
        with open("experiments/" + func_name + filename, "w", encoding="utf8") as file:
            # ueberschreibe das Dokument mit einem leeren str
            file.write("")
    except TypeError as exc:
        raise exc

def main():
    '''example implementation for using the above defined code
    '''
    # speichere Daten mittels save
    for i, element in enumerate([1, "data", "list", [1, 2]]):
        save("1_", "list.txt", i, element)
    clear("1_", "list.txt")
    for i, element in enumerate([2, "data", "list", [3, 4]]):
        save("1_", "list.txt", i, element)

# main-guard
if __name__ == "__main__":
    main()
