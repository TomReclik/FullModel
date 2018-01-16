import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.initializers import glorot_uniform, Zeros, he_normal
import tensorflow as tf

def ClassicalCNN(x_train, y_train, x_test, y_test, NOL):
    model = Sequential()

    shape = x_train[0].shape

    model.add(Conv2D(32, (3,3), input_shape=shape,
                activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Conv2D(64, (3,3), activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(512, activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros'))

    model.add(Dropout(0.5))

    model.add(Dense(NOL, kernel_initializer='he_normal', bias_initializer='zeros',
                activation='softmax'))

    return model

def CCIFC(x_train, y_train, x_test, y_test, NOL):
    """
    Convolutional Network with:
        - 1 Convolutional layers
        - 1 Inception layer
        - 1 Fully connected output layer
    """

    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(192, (5, 5), activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros')(inputs)

    tower_1 = Conv2D(160, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    tower_1 = Conv2D(96, (3, 3), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_1)

    tower_2 = Conv2D(160, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    tower_2 = Conv2D(96, (4, 4), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(96, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    maxpool = MaxPooling2D(pool_size=(3,3), strides=(2,2))(inception)

    x = Dropout(0.3)(maxpool)

    x = Conv2D(192, (5, 5), activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros')(x)

    tower_1 = Conv2D(160, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    tower_1 = Conv2D(96, (3, 3), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_1)

    tower_2 = Conv2D(160, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    tower_2 = Conv2D(96, (4, 4), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(96, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    maxpool = MaxPooling2D(pool_size=(3,3), strides=(2,2))(inception)

    x = Dropout(0.4)(maxpool)

    x = Conv2D(192, (3, 3), activation=advanced_activations.LeakyReLU(alpha=0.18),
                kernel_initializer='he_normal', bias_initializer='zeros')(x)

    tower_1 = Conv2D(100, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    tower_1 = Conv2D(64, (3, 3), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_1)

    tower_2 = Conv2D(100, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    tower_2 = Conv2D(64, (4, 4), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(64, (1, 1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    conv = Conv2D(NOL, (1,1), activation=advanced_activations.LeakyReLU(alpha=0.18),
                padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)

    x2 = Flatten()(conv)

    outputs = Dense(NOL, activation='softmax')(x2)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        # keras.callbacks.TensorBoard(log_dir=('logs/EERACN_LBFGS_'+str(len(x_train))),
        #          histogram_freq=1,
        #          write_graph=False,
        #          write_images=False)
    ]
    model.fit(x_train, y_train, batch_size=50, epochs=150, callbacks = callbacks, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=50)

    return score

def TIOSM(x_train, y_train, x_test, y_test, NOL):
    """
    Convolutional Network with:
        - 3 Inception layer
        - 1 Fully connected softmax output layer
    """

    inputs = Input(shape=(32, 32, 3))

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(10, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(64, (8, 8), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Flatten()(x)
    outputs= Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        # keras.callbacks.TensorBoard(log_dir=('logs/EERACN_LBFGS_'+str(len(x_train))),
        #          histogram_freq=1,
        #          write_graph=False,
        #          write_images=False)
    ]
    model.fit(x_train, y_train, batch_size=50, epochs=150, callbacks = callbacks, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=50)

    return score

def EERACN_Inception(x_train, y_train, x_test, y_test):
    """
    Convolutional Network like in Empirical Evaluation of Rectified Activations in Convolution
    Network
    """

    inputs = Input(shape=(32, 32, 3))

    tower_1 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(inputs)
    tower_1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(inputs)
    tower_2 = Conv2D(16, (4, 4), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), kernel_initializer='he_normal',
                bias_initializer='zeros')(inputs)
    tower_3 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    tower_1 = Conv2D(100, (1, 1), padding='valid', activation='relu')(inputs)
    tower_2 = Conv2D(100, (3, 3), padding='valid', activation='relu')(inputs)
    tower_3 = Conv2D(100, (5, 5), padding='valid', activation='relu')(inputs)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    tower_1 = Conv2D(100, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(100, (3, 3), padding='same', activation='relu')(x)
    tower_3 = Conv2D(100, (5, 5), padding='same', activation='relu')(x)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    tower_1 = Conv2D(100, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(100, (3, 3), padding='same', activation='relu')(x)
    tower_3 = Conv2D(100, (8, 8), padding='same', activation='relu')(x)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Flatten()(x)
    outputs= Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=40, epochs=60)

    score = model.evaluate(x_test, y_test, batch_size=40)

    return score

def SCFN(x_train, y_train, x_test, y_test, channels, dropout, dense):
    """
    Convolutional Network with:
        - 6 Normal convolutional layers
        - 1 Fully connected output layer
    """

    model = Sequential()

    model.add(Conv2D(channels[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(channels[0], (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[0]))

    model.add(Conv2D(channels[1], (3, 3), activation='relu'))
    model.add(Conv2D(channels[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[1]))

    model.add(Conv2D(channels[2], (3, 3), activation='relu'))
    model.add(Conv2D(channels[2], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[2]))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(dropout[3]))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def FCFN(x_train, y_train, x_test, y_test):
    """
    Convolutional Network with:
        - 4 Normal convolutional layers
        - 1 Fully connected output layer
    """

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def TCFN(x_train, y_train, x_test, y_test, channels):
    """
    Convolutional Network with:
        - 4 Normal convolutional layers
        - 1 Fully connected output layer
    """

    model = Sequential()

    model.add(Conv2D(channels[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(channels[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=50, epochs=25)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def Lenet(x_train, y_train, x_test, y_test):
    """
    The Lenet convolutional network
    """

    model = Sequential()

    model.add(Conv2D(6, (5,5), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(50, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))

    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=0)
        # keras.callbacks.TensorBoard(log_dir='logs/Lenet',
        #          histogram_freq=1,
        #          write_graph=True,
        #          write_images=False)
    ]

    model.fit(x_train, y_train, batch_size=20, epochs=100, callbacks = callbacks, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=20)

    return score

def Graham_Simple(shape, NOL):
    """
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    without sparsity
    k = 320
    """

    model = Sequential()

    model.add(Conv2D(320, (2,2), input_shape=shape, kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(320, (2,2), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # model.add(Dropout(0.5))

    for i in range(2,5):
        model.add(Conv2D(i*320, (2,2), kernel_initializer='he_normal',
                    bias_initializer='zeros'))

        model.add(BatchNormalization(axis=3))

        model.add(advanced_activations.LeakyReLU(alpha=0.3))
        model.add(Conv2D(i*320, (2,2), kernel_initializer='he_normal',
                    bias_initializer='zeros'))

        model.add(BatchNormalization(axis=3))

        model.add(advanced_activations.LeakyReLU(alpha=0.3))

        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # if i!=5:
        #     model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    return model

def Graham(shape, NOL):
    """
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    with sparsity
    k = 320
    """

    model = Sequential()

    model.add(Conv2D(320, (2,2), input_shape=shape, kernel_initializer='he_normal',
                bias_initializer='zeros'))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(320, (2,2), kernel_initializer='he_normal',
                bias_initializer='zeros'))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.5))

    for i in range(2,7):
        model.add(Conv2D(i*320, (2,2), kernel_initializer='he_normal',
                    bias_initializer='zeros'))
        model.add(advanced_activations.LeakyReLU(alpha=0.3))
        model.add(Conv2D(i*320, (2,2), kernel_initializer='he_normal',
                    bias_initializer='zeros'))
        model.add(advanced_activations.LeakyReLU(alpha=0.3))

        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        if i!=6:
            model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    return model

def EERACN(shape, NOL):
    """
    Network proposed in the paper by Bing Xu et. al.
    https://arxiv.org/pdf/1505.00853.pdf
    """

    model = Sequential()

    model.add(Conv2D(192, (5,5), input_shape=shape, kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(160, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(96, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    #model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (5,5), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3,3), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(10, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())

    model.add(AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='same'))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    return model

def EERACN_l2_simple(x_train, y_train, x_test, y_test, NOL):
    """
    Network proposed in the paper by Bing Xu et. al.
    https://arxiv.org/pdf/1505.00853.pdf
    Instead of dropout this utilizes the l2 norm regularizer
    """

    LAMBDA=0.0001

    model = Sequential()

    model.add(Conv2D(192, (5,5), input_shape=(32,32,3),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(160, (1,1),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(96, (1,1),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(192, (5,5),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(192, (1,1),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(192, (1,1),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(192, (3,3),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(192, (1,1),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(10, (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))

    model.add(AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='same'))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        keras.callbacks.TensorBoard(log_dir='logs/EERACN_l2_00001',
                 histogram_freq=1,
                 write_graph=False,
                 write_images=False)
    ]
    model.fit(x_train, y_train, batch_size=50, epochs=100, callbacks = callbacks, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def EERACN_l2_dec(x_train, y_train, x_test, y_test, NOL):
    """
    Network proposed in the paper by Bing Xu et. al.
    https://arxiv.org/pdf/1505.00853.pdf
    Instead of dropout this utilizes the l2 norm regularizer
    Increasing penalty
    """

    LAMBDA=0.0001

    model = Sequential()

    model.add(Conv2D(192, (5,5), input_shape=(32,32,3),
                kernel_regularizer=regularizers.l2(10*LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(160, (1,1),
                kernel_regularizer=regularizers.l2(10*LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(96, (1,1),
                kernel_regularizer=regularizers.l2(10*LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(192, (5,5),
                kernel_regularizer=regularizers.l2(5*LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(192, (1,1),
                kernel_regularizer=regularizers.l2(5*LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(192, (1,1),
                kernel_regularizer=regularizers.l2(5*LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(192, (3,3),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(192, (1,1),
                kernel_regularizer=regularizers.l2(LAMBDA)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))
    model.add(Conv2D(10, (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.18))

    model.add(AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='same'))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        keras.callbacks.TensorBoard(log_dir='logs/EERACN_l2_dec',
                 histogram_freq=1,
                 write_graph=False,
                 write_images=False)
    ]
    model.fit(x_train, y_train, batch_size=50, epochs=100, callbacks = callbacks, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def InceptionV3(x_train, y_train, x_test, y_test, NOL):

    inputs = Input(shape=(32, 32, 3))

    tower_1 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(inputs)
    tower_1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(inputs)
    tower_2 = Conv2D(16, (4, 4), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), kernel_initializer='he_normal',
                bias_initializer='zeros')(inputs)
    tower_3 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal',
                bias_initializer='zeros')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
