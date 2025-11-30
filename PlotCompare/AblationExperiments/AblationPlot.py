# %%
from joblib import load
# labels = ['CNN', 'LSTM', 'Transformer', 'TCN', 'Transformer-BiLSTM 并行']

# CNN 模型，实际为真实寿命
cnn_origin = load('cnn_origin')

# cnn-ggca-transform模型
B_cnn_transformer_origin = load('B_cnn_transformer_origin')
B_cnn_transformer_pre = load('B_cnn_transformer_pre')

# cnn-transform-bfm 模型
C_cnn_transformer_origin = load('C_cnn_transformer_origin')
C_cnn_transformer_pre = load('C_cnn_transformer_pre')

# cnn-transform 模型
D_cnn_transformer_origin = load('D_cnn_transformer_origin')
D_cnn_transformer_pre = load('D_cnn_transformer_pre')

# CNN-GGCA-transform-BFM 模型
A_CNN_GGCA_transform_BFM_origin = load('A_cnn_transformer_origin')
A_CNN_GGCA_transform_BFM_pre = load('A_cnn_transformer_pre')

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# ========= ① 一次性改默认字号 =========
plt.rcParams['font.size']      = 18     # 全局文字
plt.rcParams['legend.fontsize'] = 18    # 图例

# ========= ② 创建画布 =========
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)   # 拿到 fig / ax 句柄

# ========= ③ 画曲线 =========
ax.plot(cnn_origin,                label='真实寿命', color='black', linewidth=2)
ax.plot(A_CNN_GGCA_transform_BFM_pre, label='方法A', color='red',   linewidth=2)
ax.plot(B_cnn_transformer_pre,       label='方法B', color='c',     linewidth=2)
ax.plot(C_cnn_transformer_pre,       label='方法C', color='blue',  linewidth=2)
ax.plot(D_cnn_transformer_pre,       label='方法D', color='green', linewidth=2)

# ========= ④ 添加标签/标题/图例 =========
ax.set_xlabel('运行周期/10s', fontsize=18)
ax.set_ylabel('寿命百分比',  fontsize=18)
ax.set_title('Bearing1-3 预测结果', fontsize=18)
ax.legend(loc='upper right')           # 字号已由 rcParams 控制

fig.tight_layout()                     # 避免文字被裁剪

# ========= ⑤ 先保存，再显示 =========
fig.savefig('消融实验数据对比图.png', dpi=300, bbox_inches='tight')
plt.show()                             # 放在最后
