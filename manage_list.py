'''This is a programm for managing txt-files

Functions
---------
create_list()
    function for creating files containing lists of intergers
create_word_list()
    function for creating files containing lists of strings
get_list()
    function that reads a file created by one of the first functions and returning its content
    as a list
save()
    saves data in a specific format
clear()
    clears a file
main():
    example implementation for using the above defined code
'''

import os

def save(func_name, filename, data, data_1="", data_2=""):
    '''function that saves data in a specific format
    
    Parameters
    ----------
    filename : str
        file in which the data is saved
    iteration : int
        number which is saved in order to make the file readable for a human
    data : int or list
        data that is saved
    
    Exceptions
    ----------
    OSError
        file does not exist
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
    except OSError as exc:
        raise exc
    except TypeError as exc:
        raise exc


def clear(func_name, filename):
    '''function for clearing files
    Parameters
    ----------
    filename : str
        file that is to be cleared
    
    Exceptions
    ----------
    OSError
        file does not exist
    TypeError
        filename is not a str
    '''
    try:
        cwd = os.getcwd()
        if not os.path.exists(cwd + "/experiments"):
            os.makedirs(cwd + "/experiments")
        with open("experiments/" + func_name + filename, "w", encoding="utf8") as file:
            # ueberschreibe das Dokument mit einem leeren str
            file.write("")
    except OSError as exc:
        raise exc
    except TypeError as exc:
        raise exc

def main():
    '''example implementation for using the above defined code
    '''
    # erstelle eine Zahlenliste
    create_list(50)
    # lese diese Liste aus und gebe sie aus
    print(get_list("list.txt"))
    # erstelle eine Woerterliste
    create_word_list(50)
    # lese diese aus und speichere sie in word_list
    word_list = get_list("list.txt")
    # gebe die Liste aus
    print(word_list)
    # leere das Dokument
    clear("list.txt")
    # gebe das leere Dokument aus
    print(get_list("list.txt"))
    # speichere die Daten der Woerterliste mittels save
    for i, element in enumerate(word_list):
        save("list.txt", i, element)
    # gebe den Inhalt des Dokuments aus
    print(get_list("list.txt"))

# main-guard
if __name__ == "__main__":
    main()
    