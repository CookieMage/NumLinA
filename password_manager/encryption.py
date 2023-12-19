import random

SYMBOLS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöü,.-;:_#+*~1234567890!§$%&/()=?²³{[]}\^°@€|<>"
LEN_SYMBOLS = len(SYMBOLS)

def encrypt(text, password):
    seed = conv_to_int(password)
    rnd = random.Random(seed)
    shifts = []
    for e in text:
        shifts += [rnd.randint(0, LEN_SYMBOLS)]
    message = ""
    for i,e in enumerate(text):
        letter = SYMBOLS.find(e)
        letter += shifts[i]
        letter %= LEN_SYMBOLS
        letter = SYMBOLS[letter]
        message += letter
    return message

def decrypt(text, password):
    seed = conv_to_int(password)
    rnd = random.Random(seed)
    shifts = []
    for e in text:
        shifts += [rnd.randint(0, LEN_SYMBOLS)]
    message = ""
    for i,e in enumerate(text):
        letter = SYMBOLS.find(e)
        letter -= shifts[i]
        letter %= LEN_SYMBOLS
        letter = SYMBOLS[letter]
        message += letter
    return message

def conv_to_int(text):
    if isinstance(text, int):
        return text
    letter = ""
    for e in text:
        letter += str(SYMBOLS.find(e))
    return int(letter)

def main():
    password = "123"
    message = encrypt("Hallo mein Name ist Basti", password)
    print(message)
    print(decrypt(message, password))

if __name__ == "__main__":
    main()
