import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#PM10에 대해서 분류

class PredictModel:
    def loadData(self, X, y):
        count = len(X)
        column = len(X[0])
        row = len(X[0][1])

        X = np.array(X)
        y = np.array(y)
        X = X.reshape(count, column, row, 1)
        y = y.reshape(count, 1)

        X_train = X[:-4]
        y_train = y[:-4]
        X_test = X[-4:]
        y_test = y[-4:]

        return row, column, X_train, y_train, X_test, y_test


    def returnCallback(self):
        callbackList = [
            tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model/modelv1.h5',
                monitor= 'val_loss',
                save_best_only= True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor= 'val_loss',
                patience=20,
            )]

        return callbackList


    def model(self, row, column, X_train, y_train, X_test, y_test):
        input_layer = tf.keras.layers.Input(shape=(column, row, 1))

        x = tf.keras.layers.Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu', name='conv_1_2x2/1')(input_layer)
        x = tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1), name='max_pool_1_2x2/1', padding='same')(x)

        x = tf.keras.layers.Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu', name='conv_2_2x2/2')(x)
        x = tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1), name='max_pool_1_2x2/2', padding='same')(x)

        x = tf.keras.layers.Conv2D(128, (2, 2), padding='same', strides=(1, 1), activation='relu', name='conv_2_2x2/3')(x)
        x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='max_pool_1_2x2/3', padding='same')(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_3')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_4')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='dense_5')(x)
        x = tf.keras.layers.Dense(8, activation='relu', name='dense_6')(x)
        x = tf.keras.layers.Dense(1, name='output')(x)

        model = tf.keras.Model(input_layer, x)
        model.summary()
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse', optimizer= optimizer)
        model.fit(X_train, y_train, epochs= 1000, batch_size= 8, callbacks= self.returnCallback(), validation_data= (X_test, y_test))
        model.save(f'model/model.h5')
