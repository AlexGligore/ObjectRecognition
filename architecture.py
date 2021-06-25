from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, InputLayer, Dropout


def simple_classifier(classes_nr, input_shape=(128, 128, 3), batch_size=4):

    # model = Sequential(name="simple_classifier")
    # model.add(InputLayer(input_shape=input_shape, batch_size=batch_size))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dense(16, activation='softmax'))
    # model.add(Dropout(0.25))
    # model.add(Dense(1, activation='softmax'))
    # model.compile(optimizer='adam', loss='mse')

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, strides=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, strides=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_nr, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
