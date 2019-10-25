_pad = '<pad>'
unk = '<unk>'
eos = '<eos>'
sos = '<sos>'
mask = '<mask>'
_logits = '1234567890'
_punctuation = '\'(),.:;?$*=!/"\&-#_ \n'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# symbols = [_pad, unk, eos, sos, mask] + list(_logits) + list(_letters) + list(_punctuation)
symbols = list(_logits) + list(_letters) + list(_punctuation)

# symbols = set(symbols)
# print(len(symbols))

# dict 88
# dict = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13, '0': 14, 'A': 15, 'B': 16, 'C': 17, 'D': 18, 'E': 19, 'F': 20, 'G': 21, 'H': 22, 'I': 23, 'J': 24, 'K': 25, 'L': 26, 'M': 27, 'N': 28, 'O': 29, 'P': 30, 'Q': 31, 'R': 32, 'S': 33, 'T': 34, 'U': 35, 'V': 36, 'W': 37, 'X': 38, 'Y': 39, 'Z': 40, 'a': 41, 'b': 42, 'c': 43, 'd': 44, 'e': 45, 'f': 46, 'g': 47, 'h': 48, 'i': 49, 'j': 50, 'k': 51, 'l': 52, 'm': 53, 'n': 54, 'o': 55, 'p': 56, 'q': 57, 'r': 58, 's': 59, 't': 60, 'u': 61, 'v': 62, 'w': 63, 'x': 64, 'y': 65, 'z': 66, "'": 67, '(': 68, ')': 69, ',': 70, '.': 71, ':': 72, ';': 73, '?': 74, '$': 75, '*': 76, '=': 77, '!': 78, '/': 79, '"': 80, '\\': 81, '&': 82, '-': 83, '#': 84, '_': 85, ' ': 86, '\n': 87}