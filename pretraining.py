# Pre-trains a contrastive model for vocal similarity modelling
# Usage: python3 pretraining.py datapath model_dir --configs config_file
# eg: python3 pretraining.py my_data/ model1/ configs/myconfig.ini

# Config files contain the following parameters:
# augment: whether artificial vocal-accompaniment mixtures are created
# artist: whether artist-level (True) or segment-level (False) sampling is applied
# sources: whether vocal sources ('vocals') are used during pre-training
# finetune: originally set to False; finetunes a model in a given directory if filled.

import os
import sys
import configparser
from collections import Counter
import numpy as np
import scipy as sp
import sklearn
import copy
import argparse
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from concurrent.futures import ProcessPoolExecutor
import collections
import json
import random
import time
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

from cola import constants,data
from mscol import network

#@tf.function
def _prepare_example(y,z):#z,src):

  #print(y,z)
  """Creates an example (anchor-positive) for instance discrimination."""
  try:
    x = np.load(y)
  except:
    print('Exception! 0-load',y)
    x = np.zeros((80000,))
  pt = 0 
  s = int(np.random.uniform(0,64000))
  if any(augments):
    w = 0.6+0.2*np.random.random()
  else:  
    w = 1
  frames_anchors = w*x[pt+s:pt+s+16000] #scaling the accompaniment
  if any(srcs):
    try:
      x = np.load(z)
    except:
      print('Exception! 0-load')
      x = np.zeros((80000,))
    if any(augments):
      frames_anchors += (1-w**2)*x[pt+s:pt+s+16000]
  
  s = int(np.random.uniform(0,64000))
  frames_positives = x[pt+s:pt+s+16000]

  return frames_anchors, frames_positives

parser = argparse.ArgumentParser()
parser.add_argument('datapath',type=str) #directory to load the data from
parser.add_argument('model_dir',nargs='+',default=[]) #directory to store the model to
parser.add_argument('--configs',nargs='+',default='base.yml')


args = parser.parse_args()
datapath = args.datapath
model_dir = args.model_dir
configs = args.configs

assert(len(model_dir)==len(configs))

arts = []
augments = []
fts = []
srcs = []

confParser = configparser.ConfigParser()
for config in configs:
  confParser.read(config)
  sources = confParser['default']['sources']
  arts.append(bool(confParser['default']['artist'] == 'True'))
  augments.append(bool(confParser['default']['augment'] == 'True'))
  finetune = confParser['default']['finetune']
  if finetune == 'False':
    fts.append(False)
  else:
    fts.append(finetune)
  if sources != 'None':
    srcs.append(True)
  else:
    srcs.append(False)

print('artist sampling:',arts,'augmentation':,augments,'sources:',srcs,'finetuning':,fts)

train_path = datapath+'sliced_wavs'
valid_path = datapath+'sliced_wavs'

train_path_src = datapath+'/sliced_wavs_src/'
valid_path_src = datapath+'/sliced_wavs_src/'

if any(srcs):
  train_keys = np.load('train_vocal_keys.npy',allow_pickle=True)
  valid_keys = np.load('valid_vocal_keys.npy',allow_pickle=True)
  if any(augments):
    train_keys_acc = np.load('train_full_keys.npy',allow_pickle=True)
    valid_keys_acc = np.load('valid_full_keys.npy',allow_pickle=True)
  if any(arts):
    train_artist_keys = np.load('train_artname_keys.npy',allow_pickle=True)
    valid_artist_keys = np.load('valid_artname_keys.npy',allow_pickle=True)
else:
  if any(arts):
    train_artist_keys = np.load('train_full_artname_keys.npy',allow_pickle=True)
    valid_artist_keys = np.load('valid_full_artname_keys.npy',allow_pickle=True)
  
  train_keys = np.load('train_full_keys.npy',allow_pickle=True)
  valid_keys = np.load('valid_full_keys.npy',allow_pickle=True)

apptimes = np.zeros((len(train_keys),)).tolist() 
print(len(train_keys),len(apptimes))
print(len(train_artist_keys),len(valid_artist_keys))
apps = dict(zip(train_keys,apptimes))
print(len(apps))

train_numel = len(train_keys)
valid_numel = len(valid_keys)

print(train_numel,valid_numel)
stepinit = 0 
steps = 10000

contrastive_network = []
chkpts = []
for i in range(0,len(configs)):
  contrastive_network.append(network.get_contrastive_network(
      embedding_dim=512,
      temperature=0.2,
      pooling_type='max',
      similarity_type=constants.SimilarityMeasure.BILINEAR))

  #TODO: lrs?

  with tf.device('/GPU:'+str(i)):
    if fts[i]:
      print('finetuning')
      contrastive_network[i].compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],run_eagerly=True)

      contrastive_network[i].load_weights(tf.train.latest_checkpoint(fts[i])).expect_partial()
      contrastive_network[i].compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],run_eagerly=True)

    else:
      print('from scratch')
      contrastive_network[i].compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],run_eagerly=True)

    chkpts.append(tf.train.Checkpoint(contrastive_network[i]))

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

augment_prob = 0.5 

batchSize = 8192
ct = 0
indexlist = list(range(0,len(train_keys))) #shuffler here, we use the same loader for all the ini files
if any(augments):
  acc_indexlist = list(range(0,len(train_keys_acc)))

best_val_loss = []
lossmats = []
for k in range(0,len(configs)):
  best_val_loss.append(9999)
  lossmats.append([])

start_time = time.time()
for i in range(stepinit, stepinit+steps):
  

  if (i % 10 == 0):
    print('training contrastive model...')
    running_loss = np.zeros((len(configs),))
    running_sca = np.zeros((len(configs),))

    if (i == 0):
      dir_list = [train_path+'/'+train_key+'.npy' for train_key in train_keys]  #dir_list keeps all training paths   
      dir_subset = dir_list[:10*batchSize] #keeping samples for the 10 epochs
      if any(augments):
        acc_dir_list = [train_path+'/'+train_key for train_key in train_keys_acc]
        acc_dir_subset = acc_dir_list[:10*batchSize]
      if any(arts):
        art_subset = list(train_artist_keys[:10*batchSize])
        print(art_subset[23])

    else: #randomly replace 20% of entries
      random.shuffle(indexlist)
      print(indexlist[:50])

      internal_idlist = list(range(0,len(dir_subset)))
      random.shuffle(internal_idlist)
      dir_subset = [dir_subset[x] for x in internal_idlist]
      dir_list_new = [dir_list[x] for x in indexlist[:2*batchSize]] #shuffled entries
      dir_subset = dir_list_new+dir_subset[2*batchSize:] #keep the 80% of the initial entries intact!

      if any(augments): #unaligned with the base indexes
        random.shuffle(acc_indexlist)

        internal_idlist = list(range(0,len(acc_dir_subset)))
        random.shuffle(internal_idlist)
        acc_dir_subset = [acc_dir_subset[x] for x in internal_idlist]
        acc_dir_list_new = [acc_dir_list[x] for x in acc_indexlist[:2*batchSize]]
        acc_dir_subset = acc_dir_list_new+acc_dir_subset[2*batchSize:]
      
      if any(arts): #aligned w/ the base indexes
        art_subset = [art_subset[x] for x in internal_idlist]
        art_subset_new = [train_artist_keys[x] for x in indexlist[:2*batchSize]]
        art_subset = art_subset_new+art_subset[2*batchSize:]
      
      print('refreshed!')

    print(len(list(set(dir_subset)))) #number of unique elements in dir_subset (check if it drops overly...)

  internal_idlist = list(range(0,len(dir_subset)))
  random.shuffle(internal_idlist)
  valid_dirs = [dir_subset[x] for x in internal_idlist[:batchSize]]
  if any(arts):
    valid_arts = [art_subset[x] for x in internal_idlist[:batchSize]] 
    art_batchids = {req_word: [idx for idx, word in enumerate(valid_arts) if word == req_word] for req_word in set(valid_arts)}
    
  if any(augments): #is shuffled independently
    random.shuffle(acc_dir_subset)
    valid_acc_dirs = acc_dir_subset[:batchSize]

  for x in valid_dirs: #updating dict entries
    apps[x[39:-4]] += 1

  if not any(augments):
    anchor_dirs = [x[:-28]+'/full/'+x[-28:] for x in valid_dirs]
  else: #if any(augments)
    temp_dirs = [x[:-28]+'/accomp/'+x[-28:-4]+'_accomp.npy' for x in valid_dirs] #valid_acc_dirs under normal circumstances
    permuted_idxs = np.random.permutation(len(temp_dirs))[:int(augment_prob*len(temp_dirs))] #choice of elements to permute
    new_idxs = np.copy(permuted_idxs)
    np.random.shuffle(new_idxs) #shuffle those
    print(permuted_idxs[:10],new_idxs[:10])
    anchor_dirs = []
    act = 0
    for ii in range(0,len(temp_dirs)):
      if ii in permuted_idxs:
        anchor_dirs.append(temp_dirs[new_idxs[act]])
        act += 1
      else:
        anchor_dirs.append(temp_dirs[ii])
    #print('semishuffled')
    
  
  if not any(srcs):
    pos_dirs = valid_dirs
  else:
    pos_dirs = [x[:-28]+'/vocals/'+x[-28:-4]+'_vocals.npy' for x in valid_dirs]
    
  if any(arts):
    idlist = [random.choice(art_batchids[x]) for x in valid_arts]

  if (i%10 == 9):
    print(Counter(apps.values())) #keeps an archive of how often each element has been used for training

  anchors,positives = np.zeros((batchSize,16000)), np.zeros((batchSize,16000)) #1sec excerpts.
  executor = ProcessPoolExecutor(max_workers=4)

  pairs = list(executor.map(_prepare_example, anchor_dirs,pos_dirs))
  anchors = np.stack([x[0] for x in pairs])
  positives = np.stack([x[1] for x in pairs])

  for k in range(0,len(configs)):

    with tf.device('/GPU:'+str(i)):
      if arts[k]: 
        positives_ = positives[idlist,:]
      else:
        positives_ = positives

      data_ = np.concatenate((np.expand_dims(anchors,-1),np.expand_dims(positives_,-1)),axis=-1)
      data_[np.isnan(data_)] = 0 #prolly irrelevant
      evals = contrastive_network[k].fit(
        data_,batch_size = 128,
        epochs=1,
        verbose=2,shuffle=False)
      running_loss[k] += evals.history["loss"][0]
      running_sca[k] += evals.history["sparse_categorical_accuracy"][0]
      del positives_,data_
      gc.collect()
   
  curr_time = time.time()
  print('elapsed time',curr_time-start_time)
  start_time = curr_time

  del anchors, positives,pairs
  gc.collect()

  if (i % 10 == 9): #evaluate the proxy task every 10 passes
    
    for k in range(0,len(configs)):
      print('config',configs[k],'training contrastive task loss at step', i, ' ', running_loss[k]/10)
      print('config',configs[k],'training contrastive task accuracy at step', i, ' ', running_sca[k]/10)


    print('checkpoint: evaluating contrastive model(s)...')
      
    batch_comp = np.random.permutation(valid_numel)[:batchSize]
    valid_dirs = [valid_path+'/'+valid_keys[pid]+'.npy' for pid in batch_comp]
    if any(arts):
      valid_arts = [valid_artist_keys[x] for x in batch_comp]
      art_batchids = {req_word: [idx for idx, word in enumerate(valid_arts) if word == req_word] for req_word in set(valid_arts)}

    if any(augments):
      batchacc_comp = np.random.permutation(valid_numel)[:batchSize]
      valid_acc_dirs = [valid_path+'/'+valid_keys_acc[pid]+'.npy' for pid in batchacc_comp]

    if not any(augments):
      anchor_dirs = [x[:-28]+'/full/'+x[-28:] for x in valid_dirs]
    else:
      temp_dirs = [x[:-28]+'/accomp/'+x[-28:-4]+'_accomp.npy' for x in valid_dirs]
      anchor_dirs = []
      act = 0
      for ii in range(0,len(temp_dirs)):
        if ii in permuted_idxs:
          anchor_dirs.append(temp_dirs[new_idxs[act]])
          act += 1
        else:
          anchor_dirs.append(temp_dirs[ii])
      print('semishuffled')
    if not any(srcs):
      pos_dirs = valid_dirs
    else:
      pos_dirs = [x[:-28]+'/vocals/'+x[-28:-4]+'_vocals.npy' for x in valid_dirs]
    if any(arts):
      idlist = [random.choice(art_batchids[x]) for x in valid_arts]

    anchors,positives = np.zeros((batchSize,16000)), np.zeros((batchSize,16000))
    executor = ProcessPoolExecutor(max_workers=4)

    pairs = list(executor.map(_prepare_example, anchor_dirs,pos_dirs))
    anchors = np.stack([x[0] for x in pairs])
    positives = np.stack([x[1] for x in pairs])
    for k in range(0,len(configs)):
      with tf.device('/GPU:'+str(k)):
        if arts[k]:
          positives_ = positives[np.asarray(idlist)]
        else:
          positives_ = positives
        data = np.concatenate((np.expand_dims(anchors,-1),np.expand_dims(positives_,-1)),axis=-1)
        evals = contrastive_network[k].evaluate(data,batch_size=128,verbose=0)
        lossmats[k].append(evals[0])
        del positives_, data
        gc.collect()
        print('config',configs[k],'validation contrastive task loss at step', i, ' ', evals[0])
        print('config',configs[k],'validation contrastive task accuracy at step', i, ' ', evals[1])    

    del pairs,anchors, positives
    gc.collect()
    start_time = time.time()    

                         
  if (i%100 == 99):
    for k in range(0,len(configs)):
      with tf.device('/GPU:'+str(k)):
        save_path = chkpts[k].save('./'+model_dir[k]+'checkpoint')

        mean_val_loss = np.mean(np.asarray(lossmats[k]))
        print('current mean loss for config', configs[k],mean_val_loss)
        #if mean_val_loss > best_val_loss:
        #  K.set_value(contrastive_network[k].optimizer.lr, 0.5*K.get_value(contrastive_network[k].optimizer.lr)) #update lr
        #  print('reducing learning rate for config', configs[k],'to', K.get_value(contrastive_network[k].optimizer.lr))
        #else:
        #  best_val_loss = mean_val_loss #update best running average loss
        lossmats[k] = []

