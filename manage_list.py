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
import random
import string
import os


def create_list(size : int):
    '''creates txt-file containing a list of random integers between 0 and 1000

    Parameters
    ----------
    size : int
        length of the list that is to be generated
    
    Exceptions
    ----------
    ValueError
        size is smaller than 1
    TypeError
        size is not an int
    '''
    # finde den Speicherort des Skriptes und speichere diese mit Anhang "list.txt"
    try:
        if size < 1:
            raise ValueError
        script_dir = os.path.dirname(__file__)
        rel_path = "list.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        # erzeuge eine Zahlenliste der groesse size in list.txt
        with open(abs_file_path, "w",  encoding = "utf8") as file:
            for _ in range(size-1):
                file.write(str(random.randint(0, 1000))+", ")
            file.write(str(random.randint(0, 100)))
    except TypeError as exc:
        raise exc


def create_word_list(size : int):
    '''creates txt-file containing a list of str that contain 5 random characters

    Parameters
    ----------
    size : int
        length of the list that is to be generated
    
    Exceptions
    ----------
    ValueError
        size is smaller than 1
    TypeError
        size is not an int
    '''
    try:
        # finde den Speicherort des Skriptes und speichere diese mit Anhang "list.txt"
        script_dir = os.path.dirname(__file__)
        rel_path = "list.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        # erzeuge eine Woerterliste der groesse size in list.txt
        with open(abs_file_path, "w",  encoding = "utf8") as file:
            letters = string.ascii_lowercase.join("!§$%&/()=?1234567890²³{[]}+*~,.-;:_<>|@^°")
            for _ in range(size-1):
                # generiert Woerter der Laenge 5
                result_str = ''.join(random.choice(letters) for j in range(5))
                file.write(result_str+", ")
            result_str = ''.join(random.choice(letters) for j in range(5))
            # speichert die Woerter
            file.write(result_str)
    except ValueError as exc:
        raise exc
    except TypeError as exc:
        raise exc


def get_list(filename : str):
    '''function that reads a file and returns its contents

    Parameters
    ----------
    filename : str
        file that is to be read

    Returns
    -------
    list of str or int
        list containing the saved lines of the file
    
    Exceptions
    ----------
    OSError
        file does not exist
    TypeError
        filename is not a str
    '''
    try:
        # finde den Speicherort des Skriptes und speichere diese mit Anhang filename
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, filename)
        # liest list.txt ein und konvertiert den string zu einer list von integers oder strings
        with open(abs_file_path, "r",  encoding="utf8") as file:
            text = file.read()
            file_list = text.split(", ")
            # versuche alle Elemente von file_list in int zu konvertieren
            try:
                for num, obj in enumerate(file_list):
                    file_list[num] = int(obj)
            # belasse alle Elemente von file_list als string
            except ValueError:
                pass
            return file_list
    except OSError as exc:
        raise exc
    except TypeError as exc:
        raise exc


def save(filename, data, data1=""):
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
        # haenge die Parameter im Format "iteration data\n" hinten an das Dokument filename an
        with open(filename, "a", encoding="utf8") as file:
            file.write(str(data)+" "+str(data1)+"\n")
    except OSError as exc:
        raise exc
    except TypeError as exc:
        raise exc


def clear(filename):
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
        with open(filename, "w", encoding="utf8") as file:
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
    