import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import tensorflow as tf

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    print(os.path.join('data', configs['data']['filename']))
    print(configs['data']['columns'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.load_model("saved_models\\14042021-140257-e3.h5")

    x_train, y_train = data.get_train_data(
        seq_len=configs['data']['sequence_length'], 
        normalise=configs['data']['normalise']
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # predictions = model.predict_sequences_multiple(
    #     data=x_test, 
    #     window_size=configs['data']['sequence_length'], 
    #     prediction_len=configs['data']['sequence_length']
    # )

    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])

    print("x_test:",x_test,x_test.shape)

    print("y_test:",y_test,y_test.shape)

    predictions = model.predict_point_by_point(x_test)
    # predictions = model.predict_point_by_point(x_train)


    print("预测值：",predictions,predictions.shape)

    # 数据去正规化

    print(predictions,np.shape(predictions),"预测数据未正规划")
    predictions = predictions.reshape(-1,1)

    print(predictions,np.shape(predictions),"prediction")
    print(y_test,np.shape(y_test),"y_test")

    predictions = np.concatenate((predictions,predictions,predictions,predictions),axis=1)
    print(predictions,np.shape(predictions))
    y_test = np.concatenate((y_test,y_test,y_test,y_test),axis=1)
    print(y_test,np.shape(y_test))
    
    predictions = data.inverse_windows(predictions)[:,0]
    print(predictions,np.shape(predictions),"prediction")
    y_test = data.inverse_windows(y_test)[:,0]
    print(y_test,np.shape(y_test),"y_test")
    
    
    # 绘图

    # # plot_results_multiple(predictions, y_test,
    # #                       configs['data']['sequence_length'])

    plot_results(predictions, y_test)
    
    # 误差曲线
    wucha = y_test-predictions
    print(wucha,wucha.shape)
    plot_results(wucha,wucha)



if __name__ == '__main__':
    main()
