# %%
from joblib import load
# labels = ['CNN', 'LSTM', 'Transformer', 'TCN', 'Transformer-BiLSTM 并行']

# CNN 模型，实际为真实寿命
cnn_origin = load('cnn_origin')



# WOA-Attention-BiLSTM 模型
tcn_origin = load('tcn_origin')
tcn_pre = load('tcn_pre')


# CNN-LSTM 模型
cnn_lstm_origin = load('cnn_lstm_origin')
cnn_lstm_pre = load('cnn_lstm_pre')

# Transformer-BiLSTM 串行模型
transformer_bilstm_serial_origin = load('transformer_bilstm_serial_origin')
transformer_bilstm_serial_pre = load('transformer_bilstm_serial_pre')

# CNN-GGCA-transform-BFM 模型
CNN_GGCA_transform_BFM_origin = load('cnn_transformer_origin')
CNN_GGCA_transform_BFM_pre = load('cnn_transformer_pre')

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# ========= ① 一次性改默认字号 =========  # ←新增
plt.rcParams['font.size'] = 18                 # 所有文字默认 16 号
plt.rcParams['legend.fontsize'] = 18           # 图例默认 16 号





# 创建左侧图：损失
plt.figure(figsize=(14, 7), dpi=300)


plt.plot(cnn_origin, label='真实寿命', color='black', linewidth=3, linestyle='-', ) #

# plt.plot(tcn_pre, label='TCN', color='c', linewidth=3,  ) #linestyle=':'
plt.plot(tcn_pre, label='WOA-Attention-BiLSTM', color='c', linewidth=3,  ) #linestyle=':'

plt.plot(cnn_lstm_pre, label='CNN-LSTM', color='blue', linewidth=3, ) #linestyle='--'

plt.plot(transformer_bilstm_serial_pre, label=' Transformer-BiLSTM', color='green', linewidth=3, ) #linestyle='-'

plt.plot(CNN_GGCA_transform_BFM_pre, label='本文方法', color='red', linewidth=3, ) # linestyle='-.'


plt.xlabel('运行周期/10s', fontsize=18)
plt.ylabel('寿命百分比', fontsize=18)
plt.title('Bearing1-3 预测结果', fontsize=18)
# ======== ③ 改图例字号 =========
plt.legend(fontsize=18, loc='upper right')    # loc 可选

# 保存图像到当前目录，文件名为 "对比实验对比图.png"
plt.savefig('对比实验对比图.png', bbox_inches='tight')

plt.show()


