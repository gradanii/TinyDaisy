import collections


def encoder(string):
    # initialize everything
    byte_list = list(string.encode("utf-8"))
    byte_vocab = {bytes([i]).decode("utf-8", errors="ignore"): i for i in range(256)}
    inv_vocab = dict((v, k) for k, v in byte_vocab.items())
    new_token = 256

    # find the most frequent pair and add to the byte_vocab
    while len(byte_list) > 1:
        pairs = [tuple(byte_list[i : i + 2]) for i in range(len(byte_list) - 1)]
        frequency = collections.Counter(pairs)

        if frequency.most_common(1)[0][1] == 1:
            break

        most_freq_pair = frequency.most_common(1)[0][0]
        word = "".join(inv_vocab[i] for i in most_freq_pair)

        nbyte_list = []
        byte_vocab[word] = new_token
        inv_vocab[new_token] = word

        i = 0
        while i < len(byte_list):
            pair = tuple(byte_list[i : i + 2])
            if pair == most_freq_pair:
                nbyte_list.append(byte_vocab[word])
                i += 2
            else:
                nbyte_list.append(byte_list[i])
                i += 1

        new_token += 1

        byte_list = nbyte_list

    print(byte_list)


# git test
