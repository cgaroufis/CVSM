# Evaluates the pre-trained contrastive encoder in the task of artist similarity
# (directly probes the latent space)
# Usage: python3 artist_sim_testing.py datapath model_dir --vocal_mode
# datapath: data storage directory
# model_dir: directory of the pre-trained model
# --vocal_mode: evaluates on isolated vocals; if not included, evaluates on complete mixtures.

import gc
import os
import sys
import random
from collections import Counter
import numpy as np
import argparse
import pandas as pd
from cola import constants
from mscol import network
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('datapath',type=str) #directory to load the data from
parser.add_argument('model_dir',type=str) #directory to load the model from
parser.add_argument('--vocal_mode',required=False,action='store_true')

args = parser.parse_args()
datapath = args.datapath
model_dir = args.model_dir
vocal_mode = args.vocal_mode

print('experiment parameters: model directory', model_dir, 'vocals', vocal_mode)

#72
random.seed(72)
np.random.seed(72)
tf.random.set_seed(72)

nartists = 50 

df = pd.read_csv('/gpu-data/chgar/Music4All/music4all/id_information.csv',sep='\t') #music4all file, place it in the respective folder.
test_ids_ufilt = np.load('id_assignment.npy',allow_pickle=True)[9]
test_ids_filt = []

print(len(test_ids_ufilt),test_ids_ufilt[0])
for x in test_ids_ufilt:
    for n in range(0,6):
    
        if os.path.isfile(datapath+'/vocals/'+x+'_down_'+str(n)+'_vocals.npy'):
            test_ids_filt.append(x+'_down_'+str(n)+'.npy')

print(len(test_ids_filt),test_ids_filt[:10])


artnames = []
for test_id in test_ids_filt:
    cut_key = test_id[:16]
    artnames.append(df[df['id']==cut_key]['artist'].tolist()[0])

#keep a weighted random 50-artist subset

artfreqs = Counter(artnames)
ordered_artnames = list(artfreqs.keys())
ordered_artvalues = np.asarray(list(artfreqs.values()))/np.sum(np.asarray(list(artfreqs.values())))

contrastive_network = network.get_contrastive_network(
          embedding_dim=512,
          temperature=0.2,
          pooling_type='max',
          similarity_type=constants.SimilarityMeasure.BILINEAR)
contrastive_network.compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


contrastive_network.load_weights(tf.train.latest_checkpoint(model_dir)).expect_partial()
encoder = contrastive_network.embedding_model.get_layer("encoder")

inputs = tf.keras.layers.Input(shape=(16000,))
x = tf.math.l2_normalize(inputs, axis=0, epsilon=1e-9)
x = tf.signal.stft(inputs,frame_length=400,frame_step=160,fft_length=1024)
x = tf.abs(x)
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(64, x.shape[-1], 16000, 60, 7800)
x = tf.tensordot(x, linear_to_mel_weight_matrix, 1)
x = tf.clip_by_value(x,clip_value_min=1e-5,clip_value_max=1e8)
x = tf.expand_dims(tf.math.log(x),axis=-1)
print('before bug',x.shape)
outputs = encoder(x) #pretrained encoder of the CSSL model

model = tf.keras.Model(inputs, outputs)
model.compile(
         optimizer=tf.keras.optimizers.Adam(0.0005),
          loss=tf.keras.losses.CategoricalCrossentropy(),
          metrics=[tf.keras.metrics.CategoricalAccuracy()])


Nepisodes = 100
Ninstances = 50

names = []
batch_acc = 0
sim_acc = 0
avg_pos = 0

eers = []
mnrs = []
for k in range(0,Nepisodes):

    print('trial',k)

    idxs = np.arange(0,len(ordered_artnames))
    chosen_subset_ids = np.random.choice(idxs,size=nartists,replace=False,p=np.squeeze(ordered_artvalues))
    chosen_subset = []
    for i in range(0,nartists):

        chosen_subset.append(ordered_artnames[chosen_subset_ids[i]])

        test_ids_valid = []
        for ct,name in enumerate(artnames):
            if artnames[ct] in chosen_subset:
                test_ids_valid.append(ct)

    
    if vocal_mode:
        train_data = [test_ids_filt[x][:-4]+'_vocals'+test_ids_filt[x][-4:] for x in test_ids_valid]
    else:
        train_data = [test_ids_filt[x] for x in test_ids_valid]

    artist_id_labels_init = [artnames[x] for x in test_ids_valid]
    label_enc = OneHotEncoder(sparse_output=False).fit(np.reshape(np.asarray([artnames[x] for x in test_ids_valid]),(-1,1)))
    train_labels = label_enc.transform(np.reshape(np.asarray([artnames[x] for x in test_ids_valid]),(-1,1)))

    embs = np.zeros((2*Ninstances,1280))
    ctr = 0
    for i in range(0,Ninstances):
        ids = np.random.choice(np.where(train_labels[:,i] == 1)[0],2,False)
        for k in range(0,2):

            data_ = np.zeros((30,16000))
            cnt = 0
            for nn in range(0,6):
                if vocal_mode:
                    if os.path.isfile(datapath+'/vocals/'+train_data[ids[k]][:-12]+str(nn)+'_vocals.npy'):
                        x = np.load(datapath+'/vocals/'+train_data[ids[k]][:-12]+str(nn)+'_vocals.npy')
                        data_[cnt:cnt+5,:] = np.reshape(x,(5,16000))
                        cnt += 5
                else:
                    if os.path.isfile(datapath+'/vocals/'+train_data[ids[k]][:-6]+'_'+str(nn)+'_vocals.npy'):
                        x = np.load(datapath+'/full/'+train_data[ids[k]][:-6]+'_'+str(nn)+'.npy')   
                        
                        data_[cnt:cnt+5,:] = np.reshape(x,(5,16000))
                        cnt += 5

            data_ = data_[:cnt,:]

            temp = model.predict(data_,verbose=0)
            embs[i+Ninstances*k,:] = tf.math.l2_normalize(temp.mean(axis=0))
    print(embs.shape)

    similarities = np.matmul(embs,np.transpose(embs))

    target_similarities = similarities[:Ninstances,Ninstances:] #50x50

    #MNR
    pos_scores = []
    neg_scores =[]
    MNR = 0
    for i in range(0,Ninstances):
        ids = np.argsort(-target_similarities[i,:])
        MNR += np.where(ids == i)[0]
        pos_scores.append(target_similarities[i,i])
        neg_scores.append(target_similarities[i,(i+25)%50])
    
    pos_scores=np.sort(np.asarray(pos_scores))
    neg_scores=-np.sort(-np.asarray(neg_scores))

    print('EER',np.where(pos_scores > neg_scores)[0][0]/Ninstances)
    print('MNR',MNR/(Ninstances**2))
    eers.append(np.where(pos_scores > neg_scores)[0][0]/Ninstances)

    mnrs.append((MNR/(Ninstances**2)))

print('MNR stats',np.mean(np.asarray(mnrs)),np.std(np.asarray(mnrs)))
print('EER stats',np.mean(np.asarray(eers)),np.std(np.asarray(eers)))
