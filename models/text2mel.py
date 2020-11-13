from tensorflow.keras.layers import Embedding
from models.modules import *
from util.hparams import *


class Text2Mel(tf.keras.Model):
    def __init__(self):
        super(Text2Mel, self).__init__()

        # TextEnc
        self.textenc = tf.keras.Sequential([
            Embedding(symbol_length, embedding_dim),
            Conv1D(filters=2 * d, kernel_size=1, padding='same', activation='relu', kernel_initializer='he_normal'),
            Conv1D(filters=2 * d, kernel_size=1, padding='same', kernel_initializer='he_normal')
        ])

        for _ in range(2):
            for i in range(4):
                self.textenc.add(highway(filters=2 * d, kernel_size=3, padding='same', dilation_rate=3 ** i))

        for _ in range(2):
            self.textenc.add(highway(filters=2 * d, kernel_size=3, padding='same', dilation_rate=1))

        for _ in range(2):
            self.textenc.add(highway(filters=2 * d, kernel_size=1, padding='same', dilation_rate=1))

        # AudioEnc
        self.audioenc = tf.keras.Sequential([
            Conv1D(filters=d, kernel_size=1, padding='causal', activation='relu', kernel_initializer='he_normal'),
            Conv1D(filters=d, kernel_size=1, padding='causal', activation='relu', kernel_initializer='he_normal'),
            Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')
        ])

        for _ in range(2):
            for i in range(4):
                self.audioenc.add(highway(filters=d, kernel_size=3, padding='causal', dilation_rate=3 ** i))

        for _ in range(2):
            self.audioenc.add(highway(filters=d, kernel_size=3, padding='causal', dilation_rate=3))

        # AudioDec
        self.audiodec = tf.keras.Sequential([
            Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')
        ])

        for i in range(4):
            self.audiodec.add(highway(filters=d, kernel_size=3, padding='causal', dilation_rate=3 ** i))

        for _ in range(2):
            self.audiodec.add(highway(filters=d, kernel_size=3, padding='causal', dilation_rate=1))

        for _ in range(3):
            self.audiodec.add(Conv1D(filters=d, kernel_size=1, padding='causal', activation='relu', kernel_initializer='he_normal'))

        self.audiodec.add(Conv1D(filters=mel_dim, kernel_size=1, padding='causal', activation='sigmoid', kernel_initializer='he_normal'))


    def call(self, enc_input, dec_input):
        te = self.textenc(enc_input)
        key, value = tf.split(te, num_or_size_splits=2, axis=-1)
        query = self.audioenc(dec_input)
        context, alignment = attention(query, key, value)
        mel_out = self.audiodec(context)
        return mel_out, alignment
