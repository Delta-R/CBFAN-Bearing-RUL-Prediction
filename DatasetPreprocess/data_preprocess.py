# %% [markdown]
# ### 查看 Bearing1_1 文件下所有 csv文件形状

# %%
import os
import pandas as pd

# 指定包含 CSV 文件的目录路径
directory = './phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1'  # 替换为你的实际目录路径

# 获取目录下所有 CSV 文件的文件名
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 逐个读取 CSV 文件并输出其形状
for file in csv_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    # print(f'{file} 的形状是: {df.shape}')
    print(df.shape)

# %% [markdown]
# ### 读取一个CSV 文件 看看数据格式和形状

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# 读取数据
original_data = pd.read_csv('./phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/acc_00001.csv', header=None)
print(original_data.shape)
original_data.head()

# %% [markdown]
# ### 水平振动信号 可视化

# %%
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(original_data.iloc[:,-2], color='green')# 第四行为水平数据
plt.title('水平振动信号 可视化')
plt.subplot(122)
plt.plot(original_data.iloc[:,-1], color='green')# 第五列为垂直数据
plt.title('竖直振动信号 可视化')
plt.show()

# %% [markdown]
# # 处理 数据集下面所有 CSV 文件

# %% [markdown]
# ### 处理思路：
# - 采用水平方向数据：
# 水平方向的振动数据含有更多的退化信息,因此本文以
# 水平方向采集的数据作为研究对象（参考文献）
# - 提取 13 个特征

# %%
# 定义处理函数
import pandas as pd
import os
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import entropy
import nolds
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
from scipy.stats import skew

def get_a_bearings_data(folder):
    ''' 
    获取某个工况下某个轴承的全部n个csv文件中的数据，返回numpy数组
    dp:bearings_x_x的folder
    return:folder下n个csv文件中的数据，shape:[n*32768,2]=[文件个数*采样点数，通道数]
    同时提取每个文件的统计特征：均值、方差、峰值
    '''
    names = os.listdir(folder)
    is_acc = ['acc' in name for name in names] 
    names = names[:sum(is_acc)]
    files = [os.path.join(folder,f) for f in names]

    sample_data = pd.read_csv(files[0], header=None)
    print(pd.read_csv(files[0],header=None).shape)
    # Bearing1_4 的csv文件的分隔符是分号：';'
    sep = ';' if pd.read_csv(files[0],header=None).shape[-1]==1 else ','

    h = []  # 每个文件的水平信号
    v = []  # 每个文件的垂直信号
    features = []

    for f in files:
        data = pd.read_csv(f, header=None, sep=sep)
        
        h_data = data.iloc[:, -2].values
        v_data = data.iloc[:, -1].values
        
        h.append(h_data)
        v.append(v_data)
        
        # 提取信号  采用 水平方向数值信号
        signal = h_data
        # 将NumPy数组转换为列表
        signal = signal.tolist()
        # 计算每个文件的特征
        # 1. 峭度（Kurtosis）：衡量信号的尖锐程度，用于检测信号中的高频成分。
        kurt = kurtosis(signal)
        # 2. 熵值（Entropy）：衡量信号的复杂程度和随机性，用于检测信号的频谱特性。
        ent = entropy(signal)#  -inf 表示  负无穷（-inf），则表示信号的熵为无穷大。这通常意味着信号具有非常高的不确定性和复杂性，其中的值没有明显的模式或规律
        if (ent<-0.000001 ):
            ent = 0
        # 3. 分形值（Fractal Dimension）：衡量信号的自相似性和复杂度，用于分析信号的分形特征。
        fd = nolds.dfa(signal)
        # 4. 波形指标（Waveform Indicators）：峰值因子，用于分析信号的时域特征。
        peak_factor = np.max(np.abs(signal)) / np.sqrt(np.mean(np.square(signal)))
        # 5. 波形指标（Waveform Indicators）：脉冲因子，用于分析信号的时域特征。
        pulse_factor = np.max(np.abs(signal)) / np.mean(np.abs(signal))
        # 6. 波形指标（Waveform Indicators）：裕度因子，用于分析信号的时域特征。
        crest_factor = np.max(np.abs(signal)) / np.mean(np.sqrt(np.mean(np.square(signal))))
        # 7. 频谱指标（Spectral Indicators）：能量比值，用于分析信号的频域特征。
        # 计算信号的频谱
        sampling_rate = 1024  # 采样率
        freq, power_spectrum = welch(signal, fs=sampling_rate)
        # 计算峰值频率
        peak_freqs, _ = find_peaks(power_spectrum, height=np.mean(power_spectrum))  # 找到峰值
        # 计算能量比值
        total_energy = np.sum(power_spectrum)
        peak_energy = np.sum(power_spectrum[peak_freqs])
        energy_ratio = peak_energy / total_energy
        # 8. 频谱指标（Spectral Indicators）：谱线形指标，用于分析信号的频域特征。
        # 计算谱线形指标
        spectral_flatness = np.exp(np.mean(np.log(power_spectrum))) / (np.mean(power_spectrum))
        # 9. 统计特征（Statistical Features）：均值，用于描述信号的统计特性。
        mean = np.mean(signal)
        # 10. 统计特征（Statistical Features）：方差，用于描述信号的统计特性。
        variance = np.var(signal)
        # 11. 统计特征（Statistical Features）：偏度，用于描述信号的统计特性。
        skewness = skew(signal)
        # 12. 振动特征（Vibration Features）：包括峰值振动、有效值振动等，用于描述信号的振动特性。
        peak_vibration = np.max(np.abs(signal))
        # 13. 振动特征（Vibration Features）：包括峰值振动、有效值振动等，用于描述信号的振动特性。
        rms_vibration = np.sqrt(np.mean(np.square(signal)))
        feature_dict = {
            'Kurtosis': kurt,
            'Entropy': ent,
            'Fractal Dimension': fd,
            'Peak factor': peak_factor,
            'Pulse factor': pulse_factor,
            'Crest factor': crest_factor,

            'Energy ratio': energy_ratio,
            'Spectral flatness': spectral_flatness,
            'Mean': mean,
            'Variance': variance,
            'Skewness': skewness,
            'Peak vibration': peak_vibration,
            'Rms vibration': rms_vibration
        }
        features.append(feature_dict)

    H = np.concatenate(h)
    V = np.concatenate(v)
    print(H.shape,V.shape)

    # 将特征存储为DataFrame
    features_df = pd.DataFrame(features)
    
    return np.stack([H, V], axis=-1), features_df # 返回水平和垂直的信号以及提取的特征

# %% [markdown]
# ### Bearing1_1 处理

# %% [markdown]
# #### 读取 Bearing1_1  所有 csv 文件，形成一个完整的数据集

# %%
Bearing1_1_all_data, Bearing1_1_features_df = get_a_bearings_data('./phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/')
Bearing1_1_all_data.shape, Bearing1_1_features_df.shape
# (7175680, 2)表示有7175680行数据，2列数据，第一列为水平,第二列为垂直  (2803, 13)表示有2803个文件，每个文件有13个特征

# %%
from joblib import dump, load

# 保存数据
dump(Bearing1_1_features_df, 'Bearing1_1_features_df') 
Bearing1_1_features_df.head()

# %%
# 保存数据
dump(Bearing1_1_all_data, 'Bearing1_1_all_data') 
Bearing1_1_all_data.shape

# %% [markdown]
# Bearing1_1  可视化

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(Bearing1_1_all_data[:,0], color='green')
plt.title('水平振动信号 可视化')
plt.subplot(122)
plt.plot(Bearing1_1_all_data[:,1], color='green')
plt.title('竖直振动信号 可视化')
plt.show()

# %% [markdown]
# ### Bearing1_2 处理

# %% [markdown]
# #### 读取 Bearing1_2  所有 csv 文件，形成一个完整的数据集

# %%
Bearing1_2_all_data, Bearing1_2_features_df = get_a_bearings_data('./phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_2/')
Bearing1_2_all_data.shape, Bearing1_2_features_df.shape

# %%
# 保存数据
from joblib import dump, load

# 保存数据
dump(Bearing1_2_all_data, 'Bearing1_2_all_data') 
print(Bearing1_2_all_data.shape)

dump(Bearing1_2_features_df, 'Bearing1_2_features_df') 
Bearing1_2_features_df.head()

# %% [markdown]
# Bearing1_2  可视化

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(Bearing1_2_all_data[:,0], color='green')
plt.title('水平振动信号 可视化')
plt.subplot(122)
plt.plot(Bearing1_2_all_data[:,1], color='green')
plt.title('竖直振动信号 可视化')
plt.show()

# %% [markdown]
# # 全寿命数据(Full_Test_set)

# %% [markdown]
# ### Bearing1_3 处理

# %% [markdown]
# #### 读取 Bearing1_3  所有 csv 文件，形成一个完整的数据集

# %%
FUll_Bearing1_3_all_data, FUll_Bearing1_3_features_df = get_a_bearings_data('./phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set/Bearing1_3/')
FUll_Bearing1_3_all_data.shape, FUll_Bearing1_3_features_df.shape

# %%
# 保存数据
from joblib import dump, load

dump(FUll_Bearing1_3_all_data, 'FUll_Bearing1_3_all_data') 
print(FUll_Bearing1_3_all_data.shape)

dump(FUll_Bearing1_3_features_df, 'FUll_Bearing1_3_features_df') 
FUll_Bearing1_3_features_df.head()

# %%
# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(FUll_Bearing1_3_all_data[:,0], color='green')
plt.title('水平振动信号 可视化')
plt.subplot(122)
plt.plot(FUll_Bearing1_3_all_data[:,1], color='green')
plt.title('竖直振动信号 可视化')
plt.show()

# %% [markdown]
# # 截断数据(Test_set)

# %% [markdown]
# ### Bearing1_3 处理

# %% [markdown]
# #### 读取 Bearing1_3  所有 csv 文件，形成一个完整的数据集

# %%
Bearing1_3_all_data, Bearing1_3_features_df = get_a_bearings_data('./phm-ieee-2012-data-challenge-dataset-master/Test_Set/Bearing1_3/')
Bearing1_3_all_data.shape, Bearing1_3_features_df.shape

# %%
# 保存数据
from joblib import dump, load

dump(Bearing1_3_all_data, 'Bearing1_3_all_data') 
print(Bearing1_3_all_data.shape)

dump(Bearing1_3_features_df, 'Bearing1_3_features_df') 
Bearing1_3_features_df.head()

# %%
# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(Bearing1_3_all_data[:,0], color='green')
plt.title('水平振动信号 可视化')
plt.subplot(122)
plt.plot(Bearing1_3_all_data[:,1], color='green')
plt.title('竖直振动信号 可视化')
plt.show()

# %% [markdown]
# ## 从这里可以看出来  Test_set 只是 Full_Test_Set 的一部分！

# %% [markdown]
# # 预测处理思路：
# - 用Bearing1_1、Bearing1_2 来做训练， 
# - Full_Test_Set 中 Bearing1_3 来做测试！

# %% [markdown]
# 处理思路：
# 利用处理好的特征指标，结合剩余寿命标签，来制作新的数据集

# %%
import os
import pandas as pd
import re
from joblib import dump, load



# 加载数据集
# 健康指标
Bearing1_1_features_df = load('Bearing1_1_features_df')
Bearing1_2_features_df = load('Bearing1_2_features_df')
FUll_Bearing1_3_features_df = load('FUll_Bearing1_3_features_df')

# 该目录下 样本实际寿命 （10s）
Bearing1_1_total_rul = Bearing1_1_features_df.shape[0]
Bearing1_2_total_rul = Bearing1_2_features_df.shape[0]
FUll_Bearing1_3_total_rul = FUll_Bearing1_3_features_df.shape[0]
print(Bearing1_1_total_rul, Bearing1_2_total_rul, FUll_Bearing1_3_total_rul)


data_rul_1 = []
data_rul_2 = []
data_rul_3 = []

# 处理 Bearing1_1_features_df 
# 遍历文件名从1到total_rul
for i in range(1, Bearing1_1_total_rul+1):
    # print(i)
    # 计算 RUL
    rul = (Bearing1_1_total_rul- i)/Bearing1_1_total_rul
    data_rul_1.append(rul)

Bearing1_1_features_df['rul'] = data_rul_1

# 处理 Bearing1_2_features_df 
# 遍历文件名从1到total_rul
for i in range(1, Bearing1_2_total_rul+1):
    # print(i)
    # 计算 RUL
    rul = (Bearing1_2_total_rul- i)/Bearing1_2_total_rul
    data_rul_2.append(rul)

Bearing1_2_features_df['rul'] = data_rul_2

# 处理 FUll_Bearing1_3_features_df 
# 遍历文件名从1到total_rul
for i in range(1, FUll_Bearing1_3_total_rul+1):
    # print(i)
    # 计算 RUL
    rul = (FUll_Bearing1_3_total_rul- i)/FUll_Bearing1_3_total_rul
    data_rul_3.append(rul)

FUll_Bearing1_3_features_df['rul'] = data_rul_3


print(Bearing1_1_features_df.shape)
print(Bearing1_2_features_df.shape)
print(FUll_Bearing1_3_features_df.shape)
FUll_Bearing1_3_features_df.head()

# %%
data_rul_3[-1]

# %%
# 保存数据
Bearing1_1_features_df.to_csv('samples_data_Bearing1_1.csv', index=False)
Bearing1_2_features_df.to_csv('samples_data_Bearing1_2.csv', index=False)
FUll_Bearing1_3_features_df.to_csv('samples_data_FUll_Bearing1_3.csv', index=False)


