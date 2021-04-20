import os,json,time,math,numpy as np,matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import tensorflow as tf,pandas as pd, datetime as dt
from core.utils import Timer
from sklearn import metrics

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot(data,legend):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(data, label=legend)
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



time = Timer()
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
model.load_model("saved_models\\LSTM-UKF.h5")


x_train, y_train = data.get_train_data(
    seq_len=configs['data']['sequence_length'], 
    normalise=configs['data']['normalise']
)

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

'''
# predictions = model.predict_sequences_multiple(
#     data=x_test, 
#     window_size=configs['data']['sequence_length'], 
#     prediction_len=configs['data']['sequence_length']
# )

# predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])

# # plot_results_multiple(predictions, y_test,
# #                       configs['data']['sequence_length'])

'''


time.start()
print("正在预测...")
predictions = model.predict_point_by_point(x_test)
# predictions = model.predict_point_by_point(x_train)
time.stop()
print("预测结束...")

test = x_test[0].reshape((-1,49,4))
print("单步预测时间：")
time.start()
preone = model.model.predict(test) 
time.stop()


# 数据去正规化
print("数据反正规划开始...")
predictions = predictions.reshape(-1,1)

predictions = np.concatenate((predictions,predictions,predictions,predictions),axis=1)
y_test = np.concatenate((y_test,y_test,y_test,y_test),axis=1)


predictions = data.inverse_windows(predictions)[:,0]
y_test = data.inverse_windows(y_test)[:,0]
print("数据反正规划结束...")


## 指标计算
errors = np.abs(y_test-predictions)
average_errors = np.mean(errors,axis=0)
mse = metrics.mean_squared_error(predictions, y_test)
rmse = np.sqrt(mse)

print("平均误差为：",average_errors)
print("方差为：",mse)
print("标准差为：",rmse)

y_test = y_test.reshape(-1,1)
predictions = predictions.reshape(-1,1)
errors = errors.reshape(-1,1)


## 保存数据
output = np.concatenate((y_test,predictions,errors),axis=1)

save_fname = os.path.join('Output-%s.csv' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S')))

df = pd.DataFrame(output)
df.columns = ['y_test','predictions','errors']
df.to_csv(save_fname)

# 绘图
plot_results(predictions, y_test)
plot(errors,"errors")


# 结果

## 目前3层LSTM，100个神经元是最佳预测结果

## 目前3层LSTM，300个神经元也不错