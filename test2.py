import os, glob, librosa, scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from models.ssrn import SSRN
from models.modules import griffin_lim
from util.hparams import *


checkpoint_dir = './checkpoint/2'
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)
mel_list = glob.glob(os.path.join(save_dir, '*.npy'))


def test_step(mel, idx):
    mel = np.expand_dims(mel, axis=0)
    pred = model(mel)

    pred = np.squeeze(pred, axis=0)
    pred = np.transpose(pred)

    pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db
    pred = np.power(10.0, pred * 0.05)
    wav = griffin_lim(pred ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
    endpoint = librosa.effects.split(wav, frame_length=win_length, hop_length=hop_length)[0, 1]
    wav = wav[:endpoint]
    wav = wav.astype(np.float32)
    scipy.io.wavfile.write(os.path.join(save_dir, '{}.wav'.format(idx)), sample_rate, wav)


model = SSRN()
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

for i, fn in enumerate(mel_list):
    mel = np.load(fn)
    test_step(mel, i)
