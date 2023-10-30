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
        one of the arguments is not a str
    '''
    try:
        # get current directory
        cwd = os.getcwd()
        # create a directory for saving csv files if it does not exist already
        if not os.path.exists(cwd + "/experiments"):
            os.makedirs(cwd + "/experiments")
        # append parameters to the document func_name+filename using the format "data data1 data2\n"
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
        one of the arguments is not a str
    '''
    try:
        # get current directory
        cwd = os.getcwd()
        # create a directory for saving csv files if it does not exist already
        if not os.path.exists(cwd + "/experiments"):
            os.makedirs(cwd + "/experiments")
            # overwrite the contents of func_name+filename with an empty str
        with open("experiments/" + func_name + filename, "w", encoding="utf8") as file:
            file.write("")
    except TypeError as exc:
        raise exc

def main():
    '''example implementation for using the above defined code
    '''
    # save data using save
    for i, element in enumerate([1, "data", "list", [1, 2]]):
        save("1_", "list.txt", i, element)
    # clear document
    clear("1_", "list.txt")
    # save new data using save
    for i, element in enumerate([2, "data", "list", [3, 4]]):
        save("1_", "list.txt", i, element)

# main-guard
if __name__ == "__main__":
    main()
