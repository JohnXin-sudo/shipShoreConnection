import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
class DataLoader():
    """
        用来给模型加载数据和转换数据
    """

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        print("训练数据测试数据分割位置：",i_split)

        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)

        self.data_train = self.normalise_windows(self.data_train)
        self.data_test = self.normalise_windows(self.data_test)

        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''

        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        # data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1] # 输入是前 seq_len-1 个数据
        y = data_windows[:, -1, [0]] # 输出是sequence最后一个正规化的收盘价
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for _ in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]

        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window

        x = window[:-1] # 输入是前 seq_len-1 个数据
        
        y = window[-1, [0]] # 输出是sequence最后一个正规化的收盘价 
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''
        数据正规化，基值为0
        '''
        # normalised_data = []
        # window_data = [window_data] if single_window else window_data
        # for window in window_data:
        #     normalised_window = []
        #     for col_i in range(window.shape[1]): # 列
        #         normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]] # （0，i）列是base
        #         normalised_window.append(normalised_col)
        #     normalised_window = np.array(normalised_window).T # 将数组转置回原来的多维格式
        #     normalised_data.append(normalised_window)

        normalised_data = scaler.fit_transform(window_data)
        # print("normalised_data:")
        print(normalised_data,type(normalised_data),np.shape(normalised_data))
        return np.array(normalised_data) # 3维

    def inverse_windows(self, window_data):
        inverse_data = scaler.inverse_transform(window_data)
        # print("inverse_data:")
        print(inverse_data,type(inverse_data),np.shape(inverse_data))
        return np.array(inverse_data)



