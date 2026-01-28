import random
import numpy as np
import math

'''
description: 
    seperate a int list [0, end) with a ratio
param {int} image_nums
param {float} ratio
param {bool} is_random, random seperate
return {*}
author: wuxingxing
'''
def random_index(image_nums:int, ratio:float, is_random:bool=False, seed:int=None):
    arr = np.arange(image_nums)
    if seed:
        np.random.seed(seed)
    if is_random is True:
        np.random.shuffle(arr)
    split_idx = math.ceil(image_nums*ratio)
    train_data = arr[:split_idx]
    test_data = arr[split_idx:]
    return sorted(train_data), sorted(test_data)