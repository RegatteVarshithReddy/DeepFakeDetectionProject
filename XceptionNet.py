from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def Xception():

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        None, default_size=299, min_size=71, data_format='channels_last', require_flatten=False)

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(128, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Block 5 - 12
    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Block 14
    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 14 part 2
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fully Connected Layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    inputs = img_input

    # Create model
    model = Model(inputs, x, name='xception')
    # load weights
    model.load_weights(
        'D:\\Puzzles\\xception_weights_tf_dim_ordering_tf_kernels.h5')

    # Download and cache the Xception weights file
    # weights_path = get_file('xception_weights.h5',
    # WEIGHTS_PATH, cache_subdir='models')

    return model


# Training The model

#img_width, img_height = 256, 256

train_img_path = 'D:\\Puzzles\\real_vs_fake\\real-vs-fake\\train'
val_img_path = 'D:\\Puzzles\\real_vs_fake\\real-vs-fake\\valid'


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_img_path, target_size=(256, 256),
                                                    batch_size=32, class_mode="binary")

validation_generator = test_datagen.flow_from_directory(val_img_path, target_size=(256, 256),
                                                        batch_size=32, class_mode="binary")


base_model = Xception()

x = base_model.output

x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model1 = Model(inputs=base_model.inputs, outputs=predictions)

"""
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)
"""

count = 0
for layer in base_model.layers[:21]:
    layer.trainable = False
    print(count, layer.trainable)
    count += 1
print("base model")
print(len(base_model.layers))

model1.summary()

model1.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=1e-4),
               metrics=['accuracy'])

# Checkpoints
filepath = "D:\\Puzzles\\Saved_Models\\"

checkpoint = ModelCheckpoint(filepath='model_epochs{epoch:02d}_val_acc{val_accuracy:.2f}.hdf5', monitor="val_accuracy", verbose=0,
                             save_best_only=False, save_weights_only=False, mode="auto", period=2)


history = model1.fit(train_generator, epochs=10,
                     validation_data=validation_generator)


# Visualization

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()


# Saving model
model1.save("Xception_10epochs.h5")
