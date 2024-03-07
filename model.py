import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler

# 从第i个文件到第j个feature文件作为输入
inputStart = 0
inputEnd = 1

# 读取feature，存到一个df内
datas = []
for i in range(inputStart, inputEnd+1):
    file_path = f'cg_data/features/{i * 1000}.csv'
    datas.append(pd.read_csv(file_path))
df = pd.concat(datas, ignore_index=True)

# 分离特征和目标值
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = df.drop('target', axis=1)  # 特征集是除了目标列的所有列
y = df['target']  # 目标值是目标列

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 创建模型
model = Sequential()

# 添加第一层（输入层），并增加更多神经元
model.add(Dense(195, input_dim=X_train.shape[1], activation='relu'))

# 添加输出层
model.add(Dense(1, activation='linear'))

# 设置学习率
learning_rate = 0.01  # 可以尝试不同的值，例如0.01, 0.001, 0.0001等

# 创建优化器实例
optimizer = Adam(learning_rate=learning_rate)

# 编译模型时使用这个优化器
model.compile(loss='mean_squared_error', optimizer=optimizer)

# # 训练模型
# model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], validation_split=0.2)

# 训练模型
model.fit(X_train, y_train, epochs=500, batch_size=32)

# 评估模型
mse = model.evaluate(X_test, y_test)
print('均方误差:', mse)
