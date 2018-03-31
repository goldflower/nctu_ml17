# CNN
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils

img_size = 96
classes = 8
input_img = Input(shape=(img_size, img_size, 1))

aug = False

x = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='truncated_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_initializer='truncated_normal')(x)
output = Dense(classes, activation='sigmoid', kernel_initializer='truncated_normal')(x)

model = Model(input_img, output)
model.compile(optimizer='Nadam', loss='binary_crossentropy')

df = pd.DataFrame(np.load('training_origin64.npy'))
test = df.sample(frac=0.1)
train = df.drop(test.index)
train_X = train[train.columns[:-1]].values.astype('float32')/255
train_X = np.reshape(train_X, (len(train_X), img_size, img_size, 1))
train_t = np_utils.to_categorical(train[train.columns[-1]].values)
test_X = test[test.columns[:-1]].values.astype('float32')/255
test_X = np.reshape(test_X, (len(test_X), img_size, img_size, 1))
test_t = test[test.columns[-1]].values
test_t = np_utils.to_categorical(test_t)

count = {}
for line in train_t:
    count[np.argmax(line)] = count.get(np.argmax(line), 0) + 1
total = np.sum(list(count.values()))
for key in count.keys():
    count[key] = 1/count[key]*total

if aug:
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(train_X)

    history = model.fit_generator(datagen.flow(train_X, train_t, batch_size=32),
                                  steps_per_epoch=len(train_X)//32, epochs=200,
                                  validation_data=(test_X, test_t), class_weight=count)
else:
    history = model.fit(train_X, train_t, batch_size=32, epochs=200, shuffle=True,
                    validation_data=(test_X, test_t), class_weight=count)
predict = model.predict(train_X)
result = np.argmax(predict, axis=1) == np.argmax(train_t, axis=1)
print("training acc:", len(result[result==True]) / len(result))

predict = model.predict(test_X)
result = np.argmax(predict, axis=1) == np.argmax(test_t, axis=1)
print("test acc:", len(result[result==True]) / len(result))