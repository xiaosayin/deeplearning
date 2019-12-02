from Data import data_get
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

SegTrain = np.zeros((465,100,100,100))       # mask信息
VoxelTrain = np.zeros((465,100,100,100))     # picture 100x100x100

VoxelTest = np.zeros((117,100,100,100))
SegTest = np.zeros((117,100,100,100))

label, VoxelTrain,SegTrain, VoxelTest,SegTest = data_get(SegTrain,VoxelTrain,VoxelTest,SegTest)   # label为其标签

x_train = VoxelTrain * SegTrain            # train data
x_test = VoxelTest * SegTest               # test data


# define model
model = Sequential()   # 基础层

# 添加3d卷积
model.add(Convolution3D(
        32,
        kernel_dim1= 5, # depth
        kernel_dim2= 5 , # rows
        kernel_dim3= 5, # cols
        input_shape=(100, 100, 100,1),
        activation='relu',
        #data_format='channels_first'
    ))

# 池化
model.add(MaxPooling3D(pool_size=(5, 5, 5)))

# 防止过拟合
model.add(Dropout(0.5))

# 卷积层到全连接层的过渡
model.add(Flatten())

# 全连接层
model.add(Dense(128, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, init='normal'))

model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])   #loss = categorical_crossentropy

x_train = x_train[:,:,:,:,np.newaxis]
x_test = x_test[:,:,:,:,np.newaxis]

label = label[:,np.newaxis]
print(x_train.shape)

model.fit(x_train, label, epochs=1, batch_size=16)

classes = model.predict(x_test, batch_size=16)

# 保存预测文件，分隔符为,
np.savetxt('new.csv', classes, delimiter = ',')