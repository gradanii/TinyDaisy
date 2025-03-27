import collections

def encoder(string):
    byte_list = [byte for byte in bytes(string, 'utf-8')]
    pairs = [tuple(byte_list[i:i+2]) for i in range(0, len(byte_list))]
    print(byte_list)
    print(pairs)
    frequency = dict(collections.Counter(pairs))
    print(frequency)

encoder('apples were applied')
