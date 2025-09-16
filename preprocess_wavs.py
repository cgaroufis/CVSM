# Normalizes the wav files, computes accompaniments from song-vocal pairs, and replaces low-amplitude with silence.
# Usage: python3 preprocess_wavs.py audio_path

import os
import sys
import numpy as np
import librosa
import tensorflow as tf

base_path = sys.argv[1]
vocal_path = base_path+'vocals_ds/' 
target_path = base_path+'/sliced_wavs/'

ct = 0

for _file in os.listdir(base_path):

    song_full,sr = librosa.load(base_path+'/'+_file,16000)
    song_vocal,sr = librosa.load(vocal_path+'/'+_file,16000)
    if len(song_full) < 480000:
        song_full = np.concatenate((song_full,np.zeros((480000-len(song_full),))))
        song_vocal = np.concatenate((song_vocal,np.zeros((480000-len(song_vocal),))))
    if len(song_full) > 480000:
        song_full = song_full[:480000]
        song_vocal = song_vocal[:480000]
    song_full = np.reshape(song_full,(6,80000))
    song_vocal = np.reshape(song_vocal,(6,80000)) 
    song_vocal = song_vocal * np.expand_dims((np.mean(np.abs(song_vocal),axis=1) > 0.01),axis=-1)
    song_accomp = song_full - song_vocal
    for kk in range(0,6): 
        if np.mean(np.abs(song_full[kk,:])) > 0:
            z = tf.math.l2_normalize(song_full[kk,:], epsilon=1e-9)
            np.save(target_path+_file[:-4]+'_'+str(kk)+'.npy',z)
        if np.mean(np.abs(song_vocal[kk,:])) > 0:
            z = tf.math.l2_normalize(song_vocal[kk,:], epsilon=1e-9)
            np.save(target_path+_file[:-4]+'_'+str(kk)+'vocals.npy',z)
        if np.mean(np.abs(song_accomp[kk,:])) > 0:
            z = tf.math.l2_normalize(song_accomp[kk,:], epsilon=1e-9)
            np.save(target_path+_file[:-4]+'_'+str(kk)+'accomp.npy',z)

    ct += 1

    print('done with', ct, 'files; ', len(os.listdir(vocal_path)),'files remaining')
