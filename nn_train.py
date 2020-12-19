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

