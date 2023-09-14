import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# class BusRecord():
#     def __init__(self):


class NYBDataset(Dataset):
    def __init__(self, data_dir, usernum, order=0, model_name='LSTM'):
        self.data_dir = data_dir
        self.usernum = usernum

        self.raw_data = pd.read_csv(self.data_dir, on_bad_lines='skip')
        print("Read Data Successful!")

        self.raw_data.drop(['DirectionRef', 'PublishedLineName', 'OriginName', 'OriginLat', 'OriginLong', 'DestinationName',
                   'DestinationLat', 'DestinationLong', 'NextStopPointName', 'ArrivalProximityText', 'DistanceFromStop',
                   'ExpectedArrivalTime', 'ScheduledArrivalTime'], axis=1, inplace=True)
        self.raw_data = self.raw_data.drop_duplicates() 

        valuecount = self.raw_data['VehicleRef'].value_counts()
        self.BusList = valuecount.index.tolist()
        self.BusList = self.BusList[order]
        print("user id: {}".format(self.BusList))

        self.selected_data = self.raw_data.drop(self.raw_data[[False if i in self.BusList else True for i in self.raw_data.VehicleRef]].index)
        print("selected_data shape: {}".format(self.selected_data.shape))

        self.selected_data = np.array(self.selected_data)

        if model_name == 'LSTM':
            columns_to_normalize = [2, 3]
            scaler = MinMaxScaler()
            self.selected_data[:, columns_to_normalize] = scaler.fit_transform(self.selected_data[:, columns_to_normalize])

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, index):
        return self.selected_data[index]

    def show(self):
        print(self.selected_data)


class TokyoCheckinDataset(Dataset):
    def __init__(self):
        def transfer_time(year, month, day, timestamp):
            month_list = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07',
                          'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            str1 = year + '-' + month_list[month] + '-' + day + ' ' + timestamp
            return str1

        self.raw_data = list()
        with open('data/dataset_tsmc2014/dataset_TSMC2014_TKY.txt', encoding='UTF=8', errors='ignore') as fid:
            for i, line in enumerate(fid):
                infolist = line.strip().split()
                uid = infolist[0]
                tim = transfer_time(infolist[-1], infolist[-5], infolist[-4], infolist[-3])
                lat = float(infolist[-9])
                lnt = float(infolist[-8])
                self.raw_data.append([tim, uid, lat, lnt])
        print("Read Tokyo Check-in data successful!")

        self.raw_data_pd = pd.DataFrame(self.raw_data)
        self.raw_data_pd = self.raw_data_pd.drop_duplicates()
        self.raw_data_pd = self.raw_data_pd.drop(self.raw_data_pd[[False if i == '822' else True for i in self.raw_data_pd[1]]].index)
        print("data.shape: {}".format(self.raw_data_pd.shape))

        self.raw_data_np = np.array(self.raw_data_pd)

        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        self.raw_data_np[:, columns_to_normalize] = scaler.fit_transform(self.raw_data_np[:, columns_to_normalize])

    def __len__(self):
        return len(self.raw_data_np)

    def __getitem__(self, index):
        return self.raw_data_np[index]


class GowallaCheckinDataset(Dataset):
    def __init__(self):
        def transfer_time(timestamp):
            year = timestamp[0:4]
            month = timestamp[5:7]
            day = timestamp[8:10]
            hour = timestamp[11:13]
            minute = timestamp[14:16]
            second = timestamp[17:19]
            str = year + '-' + month + '-' + day + ' ' + hour + ':' + minute + ':' + second
            return str

        self.raw_data = list()
        with open('data/loc-gowalla_totalCheckins/Gowalla_totalCheckins.txt', encoding='UTF-8', errors='ignore') as fid:
            for i, line in enumerate(fid):
                infolist = line.strip().split()
                uid = infolist[0]
                tim = transfer_time(infolist[1])
                lat = float(infolist[2])
                lnt = float(infolist[3])
                self.raw_data.append([tim, uid, lat, lnt])
        print("Read Gowalla Check-in data successful!")

        self.raw_data_pd = pd.DataFrame(self.raw_data)
        self.raw_data_pd = self.raw_data_pd.drop_duplicates()
        self.raw_data_pd = self.raw_data_pd.drop(self.raw_data_pd[[False if i == '10971' else True for i in self.raw_data_pd[1]]].index)
        print("data.shape: {}".format(self.raw_data_pd.shape))

        self.raw_data_np = np.flipud(np.array(self.raw_data_pd))

        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        self.raw_data_np[:, columns_to_normalize] = scaler.fit_transform(self.raw_data_np[:, columns_to_normalize])

    def __len__(self):
        return len(self.raw_data_np)

    def __getitem__(self, index):
        return self.raw_data_np[index]





