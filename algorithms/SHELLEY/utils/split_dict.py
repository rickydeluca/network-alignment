import random

def shuffle_and_split(dictionary):
    """
    Shuffle and split in two halfs a dictionary.
    """
    keys = list(dictionary.keys())
    random.shuffle(keys)
   
    midpoint = len(keys) // 2

    first_half_keys = keys[:midpoint]
    second_half_keys = keys[midpoint:]

    first_half = {key: dictionary[key] for key in first_half_keys}
    second_half = {key: dictionary[key] for key in second_half_keys}

    return first_half, second_half

