from tensorflow.keras.layers import Conv1DTranspose
from models.modules import *
from util.hparams import *


class SSRN(tf.keras.Model):
    def __init__(self):
        super(SSRN, self).__init__()
        self.model = tf.keras.Sequential([
            Conv1D(filters=c, kernel_size=1, padding='same', kernel_initializer='he_normal'),
            highway(filters=c, kernel_size=3, padding='same', dilation_rate=1),
            highway(filters=c, kernel_size=3, padding='same', dilation_rate=3)

        ])

        for _ in range(2):
            self.model.add(Conv1DTranspose(filters=c, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))
            self.model.add(highway(filters=c, kernel_size=3, padding='same', dilation_rate=1))
            self.model.add(highway(filters=c, kernel_size=3, padding='same', dilation_rate=3))

        self.model.add(Conv1D(filters=2*c, kernel_size=1, padding='same', kernel_initializer='he_normal'))

        for _ in range(2):
            self.model.add(highway(filters=2*c, kernel_size=3, padding='same', dilation_rate=1))

        self.model.add(Conv1D(filters=f, kernel_size=1, padding='same', kernel_initializer='he_normal'))

        for _ in range(2):
            self.model.add(Conv1D(filters=f, kernel_size=1, padding='same', activation='relu', kernel_initializer='he_normal'))

        self.model.add(Conv1D(filters=f, kernel_size=1, padding='same', activation='sigmoid', kernel_initializer='he_normal'))

    def call(self, mel_input):
        mel_output = self.model(mel_input)
        return mel_output
