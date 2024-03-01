import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# 从第i个文件到第j个fi_i文件作为输入
inputStart = 0
inputEnd = 8

data = []

for i in range(inputStart, inputEnd+1):
    file_path = f'cg_data/fi_i/{i * 1000}.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(';')
            features = []
            for part in parts[:-2]:
                features.extend([float(value) for value in part.split(',') if value.strip()])
            target = float(parts[-2].strip()) if parts[-2].strip() else None
            data.append(features + [target])

# 创建DataFrame
df = pd.DataFrame(data)

# 设置列标题
column_titles = [str(i) for i in range(len(df.columns) - 1)] + ['target']
df.columns = column_titles

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
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# # 添加额外的隐藏层
# model.add(Dense(64, activation='relu')) # 新增加的隐藏层

# # 添加原有的隐藏层，并调整神经元数量
# model.add(Dense(32, activation='relu'))

# 添加输出层
model.add(Dense(1, activation='linear'))

# 设置学习率
learning_rate = 0.001  # 可以尝试不同的值，例如0.01, 0.001, 0.0001等

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
