# convolution autoencoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import pandas as pd
import numpy as np 
import scipy.misc as misc 

img_size = int(input())
input_img = Input(shape=(img_size, img_size, 1))

x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='Nadam', loss='binary_crossentropy')

# for convolution
df = pd.DataFrame(np.load('training_grey_{}.npy'.format(img_size)))
test = df.sample(frac=0.1)
train = df.drop(test.index)
train_X = train[train.columns[:-1]].values.astype('float32')/255
train_X = np.reshape(train_X, (len(train_X), img_size, img_size, 1))
train_t = train[train.columns[-1]].values
test_X = test[test.columns[:-1]].values.astype('float32')/255
test_X = np.reshape(test_X, (len(test_X), img_size, img_size, 1))
test_t = test[test.columns[-1]].values

# 並不直接去train encoder/decoder, 而是train autoencoder, 把其中的結果拿出來作為encoder/decoder
history = autoencoder.fit(train_X, train_X, epochs=100, batch_size=32,
                          shuffle=True, validation_data=(test_X, test_X))

encoder.save('conv_encoder_grey_{}'.format(img_size))
autoencoder.save('conv_autoencoder_grey_{}'.format(img_size))