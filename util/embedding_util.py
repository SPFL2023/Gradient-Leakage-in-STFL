import numpy as np
import pandas as pd
import random

'''
dataset: nycb mta_1706
dimension=(lat 361 + lnt 427 + time 189) 977 788
lat 40.502879 ~ 40.932657
lnt -74.252339 ~ -73.701414
'''


def find_lat_candidate(lat):
    lat_candidate_set = np.load('util/lat_candidate_set.npy', allow_pickle=True)
    vec = abs(lat_candidate_set - lat)
    return np.argmin(vec), lat_candidate_set[np.argmin(vec)], len(lat_candidate_set)


def find_lnt_candidate(lnt):
    lnt_candidate_set = np.load('util/lnt_candidate_set.npy', allow_pickle=True)
    vec = abs(lnt_candidate_set - lnt)
    return np.argmin(vec), lnt_candidate_set[np.argmin(vec)], len(lnt_candidate_set)


def location_embedding(location):
    location_code_list = []
    for i in range(len(location)):
        lat = location[i][0]
        lnt = location[i][1]
        lat_idx, _, lat_can_len = find_lat_candidate(lat)
        lnt_idx, _, lnt_can_len = find_lnt_candidate(lnt)

        lat_code = np.zeros(lat_can_len)
        lat_code[lat_idx] = 1

        lnt_code = np.zeros(lnt_can_len)
        lnt_code[lnt_idx] = 1

        location_code = np.concatenate((lat_code, lnt_code))
        location_code_list.append(location_code.tolist())

    location_code_list_np = np.array(location_code_list).astype(np.float32)
    return location_code_list_np


# for time
def one_hot_encoding(raw_value, max_value, interval):
    t = int((raw_value/max_value) * interval % interval)
    one_hot_code = np.zeros(interval)
    one_hot_code[t] = 1
    return one_hot_code


def location_time_embedding(location_code, time):
    time_code_list = None
    for i in range(len(time)):
        time_code = np.array([])
        parsed_time = pd.to_datetime(time[i].item())
        year = parsed_time.year
        time_code = np.concatenate([time_code, one_hot_encoding(year - 2016, 2, 2)])
        month = parsed_time.month
        time_code = np.concatenate([time_code, one_hot_encoding(month, 12, 12)])
        day = parsed_time.day
        time_code = np.concatenate([time_code, one_hot_encoding(day, 31, 31)])
        hour = parsed_time.hour
        time_code = np.concatenate([time_code, one_hot_encoding(hour, 24, 24)])
        minute = parsed_time.minute
        time_code = np.concatenate([time_code, one_hot_encoding(minute, 60, 60)])
        second = parsed_time.second
        time_code = np.concatenate([time_code, one_hot_encoding(second, 60, 60)])
        time_code = time_code.reshape((1, len(time_code)))
        if i == 0:
            time_code_list = time_code
        else:
            time_code_list = np.concatenate((time_code_list, time_code), axis=0)
    all_code = np.concatenate((location_code, time_code_list), axis=1)
    return all_code


def time_embedding(time):
    time_code_list = None
    for i in range(len(time)):
        time_code = np.array([])
        parsed_time = pd.to_datetime(time[i].item())
        year = parsed_time.year
        time_code = np.concatenate([time_code, one_hot_encoding(year - 2016, 2, 2)])
        month = parsed_time.month
        time_code = np.concatenate([time_code, one_hot_encoding(month, 12, 12)])
        day = parsed_time.day
        time_code = np.concatenate([time_code, one_hot_encoding(day, 31, 31)])
        hour = parsed_time.hour
        time_code = np.concatenate([time_code, one_hot_encoding(hour, 24, 24)])
        minute = parsed_time.minute
        time_code = np.concatenate([time_code, one_hot_encoding(minute, 60, 60)])
        second = parsed_time.second
        time_code = np.concatenate([time_code, one_hot_encoding(second, 60, 60)])
        time_code = time_code.reshape((1, len(time_code)))
        if i == 0:
            time_code_list = time_code
        else:
            time_code_list = np.concatenate((time_code_list, time_code), axis=0)
    return time_code_list


def inv_embedding_location(code_emb):
    code_np = code_emb.reshape(788)
    lat_code = code_np[0:361]
    lnt_code = code_np[361:788]

    lat_candidate_set = np.load('util/lat_candidate_set.npy', allow_pickle=True)
    lnt_candidate_set = np.load('util/lnt_candidate_set.npy', allow_pickle=True)

    lat = lat_candidate_set[np.argmax(lat_code)]
    lnt = lnt_candidate_set[np.argmax(lnt_code)]

    return [lat, lnt]
