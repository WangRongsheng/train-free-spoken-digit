## 数据

> 原始数据：[https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

`speech_number0-9_dataset`为数据集文件，其文件命名格式为`{digitLabel}_{speakerName}_{index}.wav`

例如：`7_jackson_32.wav`这个数据文件为`jackson`提供的数字`7`的音频数据

## 预处理

`dataset_cg`：

```python
# -*- coding: UTF-8 -*-
import os

#获得文件夹下文件名列表
path=r"./speech_number0-9_dataset/"
file_list=os.listdir(path)

#选择要重命名的文件夹路径
os.chdir(path)

#将文件名中的第一个'_'用'.'替代
count = 0
for file in file_list:
  count = count +1
  print(count)
  os.rename(file,file.replace('_','.',1))
```

## 训练

训练平台为：[Edgeimpulse](https://studio.edgeimpulse.com/)

> `Create impulse`中的**Windows size**和**Windows increase**务必设为100和50！

训练的神经网络代码`nn_train.py`：

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape, BatchNormalization,Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm

# Sequential模型
model = Sequential()
# 添加输入层
model.add(InputLayer(input_shape=(X_train.shape[1], ), name='x_input'))
model.add(Reshape((int(X_train.shape[1] / 13), 13, 1),input_shape=(X_train.shape[1], )))
# 添加卷积层
model.add(Conv2D(10,kernel_size=5, activation='relu',padding='same',kernel_constraint=MaxNorm(3)))
# 添加平均池化层
model.add(AveragePooling2D(pool_size=2, padding='same'))
# 添加卷积层和平均池化层
model.add(Conv2D(5,kernel_size=5, activation='relu',padding='same',kernel_constraint=MaxNorm(3)))
model.add(AveragePooling2D(pool_size=2, padding='same'))
# 添加flatten层，拉伸铺平
model.add(Flatten())
# 添加全连接层
model.add(Dense(classes, activation='softmax', name='y_pred',kernel_constraint=MaxNorm(3)))

# 采用Adam优化器
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999)
# 网络编码
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# 模型训练
model.fit(X_train,Y_train, batch_size=32, epochs=9,validation_data=(X_test,Y_test), verbose=2)
```

## 训练结果

![result](https://s3.ax1x.com/2020/12/19/rUUtiD.png)

## 参考

[新冠肺炎咳嗽实时检测 - 人工智能垂直领域工程项目案例分享](https://www.bilibili.com/video/BV1sK411N7iC)
