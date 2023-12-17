import encryption as crypt

def encode_data(platform, account_user_name, account_mail, account_password, seed):
    text_0 = platform + ":"
    text_1 = " Username: " + account_user_name + " Mail: " + account_mail
    text_1 += " PW: " + account_password
    text_0 = crypt.encrypt(text_0, seed)
    text_1 = crypt.encrypt(text_1, seed)
    text_1 = crypt.encrypt(text_1, platform)
    return text_0 + text_1

def add_data(text):
    with open("password_manager\\passwords.txt", "a", encoding="utf8") as f:
        f.write(text + "\n")

def read_data():
    with open("password_manager\\passwords.txt", "r", encoding="utf8") as f:
        data = f.readlines()
    return data

def decode_data(data, seed):
    text = crypt.decrypt(data, seed)
    return text

def add_entry():
    platform = input("Für welche Plattform soll ein Eintrag erstellt werden?   ")
    account_user_name = input("Wie lautet der Account-Name?                             ")
    account_mail = input("Wie lautet die Account-Mail?                             ")
    account_password = input("Wie lautet das Account-Paswort?                          ")
    seed = input("Wie lautet der Verschlüsselungsseed?                     ")
    encoded = encode_data(platform, account_user_name, account_mail, account_password, seed)
    add_data(encoded)

def search_entry(entry, seed):
    entry = str(entry).lower()
    data = []
    headings = []
    data = read_data()
    for e in data:
        headings += [decode_data(e, seed).split(":")[0]]

    no_hit = []
    for i,e in enumerate(headings):
        if entry not in e.lower():
            no_hit += [i]
    for e in reversed(no_hit):
        headings.pop(e)
        data.pop(e)

    for i,e in enumerate(headings):
        data[i] = data[i][len(e)+1:]

    while len(data) > 1:
        text = "The following entries have been found:\n"
        for i,e in enumerate(headings):
            text += f"    {i}) {e}\n"
        text += "Please choose one of these entries.\n"
        choice = input(text)
        try:
            data = [data[int(choice)]]
        except (ValueError, IndexError):
            for i,e in enumerate(headings):
                if e.lower() == choice.lower():
                    data = [data[i]]
                    headings = [headings[i]]
                    break
            if len(data) > 1:
                print("The platform you meant could not be identified. Please try again.")

    if len(headings) == 0:
        raise IndexError

    return headings[0], data[0]



def decode_entry(entry, seed):
    try:
        platform, data = search_entry(entry, seed)
    except IndexError as exc:
        raise exc
    data = decode_data(data, platform)
    data = decode_data(data, seed)
    data = data.split()
    text = platform + ":"
    for i in range(0, 6, 2):
        text += "\n    " + data[i] + " " + data[i+1]
    text = "\n" + text[:-1] + "\n"
    print(text)




def main():
    directions = "If you want to quit, type 'quit'. Do you want to 'add' or 'read' an entry?\n"
    mode = input(directions).lower()
    while mode != "quit":
        if mode == 'read':
            platform = input("Which entry are you looking for?\n")
            seed = input("What is your key?\n")
            try:
                decode_entry(platform, seed)
            except IndexError:
                print("\nThe platform you are looking for could not be found. Please try again.")
            mode = input(directions).lower()
        elif mode == 'add':
            platform = input("platform: ")
            account_user_name = input("user_name: ")
            account_mail = input("mail: ")
            account_password = input("password: ")
            seed = input("key: ")
            text = encode_data(platform, account_user_name, account_mail, account_password, seed)
            add_data(text)
            mode = input(directions).lower()
        else:
            mode = input("Command could not be found. " + directions).lower()

if __name__ == "__main__":
    main()
