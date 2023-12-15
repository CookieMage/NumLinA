import encryption as crypt

def encode_data(platform, account_user_name, account_mail, account_password, seed):
    text_0 = platform + ":\n"
    text_1 = "    Username: " + account_user_name + "\n    Mail: " + account_mail
    text_1 += "\n    PW: " + account_password
    temp_seed = seed + crypt.conv_to_int(platform)
    text_0 = crypt.encrypt(text_0, seed)
    text_1 = crypt.encrypt(text_1, temp_seed)
    return text_0 + text_1

def add_data(text):
    with open("password_manager\\passwords.txt", "a", encoding="utf8") as f:
        f.write(text + "\n\n")

def read_data():
    with open("password_manager\\passwords.txt", "r", encoding="utf8") as f:
        data = f.read()
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


def decode_entry(entry, seed):


def print_entry(entry):





def main():
    seed = int(input("seed: "))
    platform = input("platform: ")
    account_user_name = input("user_name: ")
    account_mail = input("mail: ")
    account_password = input("password: ")
    text = encode_data(platform, account_user_name, account_mail, account_password, seed)
    add_data(text)

if __name__ == "__main__":
    main()

#Minecraft

#password + "Minecraft"
#Jede reelle Zahl ist äquivalent zu einer Äquivalenzklasse einer rationalen Cauchy-Folge