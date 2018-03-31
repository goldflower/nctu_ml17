import pandas as pd
import numpy as np
import keras
np.random.seed(0)

# autoencoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

img_size = int(input())
input_img = Input(shape=(img_size**2,))

x = Dense(512, activation='relu', kernel_initializer='truncated_normal')(input_img)
x = Dense(256, activation='relu', kernel_initializer='truncated_normal')(x)
x = Dense(128, activation='relu', kernel_initializer='truncated_normal')(x)
x = Dense(64, activation='relu', kernel_initializer='truncated_normal')(x)
#x = Dense(32, activation='relu', kernel_initializer='truncated_normal', kernel_regularizer='l2')(x)
encoded = Dense(32, activation='relu', kernel_initializer='truncated_normal')(input_img)

#x = Dense(32, activation='relu', kernel_initializer='truncated_normal')(encoded)
x = Dense(64, activation='relu', kernel_initializer='truncated_normal')(x)
x = Dense(128, activation='relu', kernel_initializer='truncated_normal')(x)
x = Dense(256, activation='relu', kernel_initializer='truncated_normal')(x)
x = Dense(512, activation='relu', kernel_initializer='truncated_normal')(x)
decoded = Dense(img_size**2, activation='sigmoid', kernel_initializer='truncated_normal')(encoded)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

# for normal
df = pd.DataFrame(np.load('training_grey_{}.npy'.format(img_size)))
test = df.sample(frac=0.1)
train = df.drop(test.index)
train_X = train[train.columns[:-1]].values.astype('float32')/255
train_t = train[train.columns[-1]].values
test_X = test[test.columns[:-1]].values.astype('float32')/255
test_t = test[test.columns[-1]].values
# 並不直接去train encoder/decoder, 而是train autoencoder, 把其中的結果拿出來作為encoder/decoder
history = autoencoder.fit(train_X, train_X, epochs=1000, batch_size=64,
                          shuffle=True, validation_data=(test_X, test_X))

encoder.save('encoder_grey_{}'.format(img_size))
autoencoder.save('autoencoder_grey_{}'.format(img_size))

# encode and decode some digits
encoded_imgs = encoder.predict(test_X)
decoded_imgs = autoencoder.predict(test_X)

#畫一些圖看一下效果
import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(100, 10))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_X[i].reshape(img_size, img_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(img_size, img_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('results_{}.png'.format(img_size))

