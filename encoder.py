def encoder(string):
    chars = list(dict.fromkeys(string))
    char_dict = {key: i for i, key in enumerate(chars)}
    char_list = list(string)

    while len(char_list) > 1:
        pairs = []

        for i in range(0, len(char_list)):
            pair = char_list[i : i + 2]
            if len(pair) == 2:
                pairs.append(pair)

        unique_pairs = sorted(pairs)
        frequency = {}

        for i in unique_pairs:
            freq = pairs.count(i)
            frequency["".join(i)] = freq

        frequency = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))

        max_val = max(char_dict.items(), key=lambda x: x[1])[1]

        most_freq_pair = list(frequency.keys())[0]

        char_dict[most_freq_pair] = max_val + 1

        new_list = []
        i = 0

        while i < len(char_list):
            pair = "".join(char_list[i : i + 2])
            if pair == most_freq_pair:
                new_list.append(most_freq_pair)
                i += 2
            else:
                new_list.append(char_list[i])
                i += 1

        char_list = new_list

    print(char_dict)
    print(char_list)


encoder("appple")
