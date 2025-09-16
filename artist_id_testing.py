# Evaluates the pre-trained contrastive encoder im the task of artist identification
# Usage: python3 artist_id_testing.py datapath model_dir/ [--perc perc] [--vocal_mode]
# datapath: data storage directory (where the sliced wav files are stored)
# model_dir: directory of the pre-trained model
# perc: percentage of data used for training ([0,1] range, to reproduce low-resource experiments) -- defaults to 1.
# --vocal_mode: evaluates on isolated vocals; if not included, evaluates on complete mixtures.

import random
import gc
import os
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from cola import constants
from mscol import network 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import tensorflow as tf


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('datapath',type=str) #directory to load the data from
parser.add_argument('model_dir',type=str) #directory to load the model from
parser.add_argument('--perc',type=float,default=1)
parser.add_argument('--vocal_mode',required=False,action='store_true')

args = parser.parse_args()
datapath = args.datapath
model_dir = args.model_dir
perc = args.perc
vocal_mode = args.vocal_mode

print('experiment parameters: model directory', model_dir, 'vocals', vocal_mode, 'data percentage', perc)

Nreps = 5
nartists = 50 

metrics = []

df = pd.read_csv('/gpu-data/chgar/Music4All/music4all/id_information.csv',sep='\t')
test_ids_ufilt = np.load('id_assignment.npy',allow_pickle=True)[9]
test_ids_filt = []

print(len(test_ids_ufilt),test_ids_ufilt[0])
for x in test_ids_ufilt:
    for n in range(0,6):
        if os.path.isfile(datapath+'vocals/'+x+'_down_'+str(n)+'_vocals.npy'):
            test_ids_filt.append(x+'_down_'+str(n)+'.npy')

print(len(test_ids_filt),test_ids_filt[:10])

#retrieve artist labels from ids

artnames = []
for test_id in test_ids_filt:
    cut_key = test_id[:16]
    artnames.append(df[df['id']==cut_key]['artist'].tolist()[0])

#keep a weighted random 50-artist subset

artfreqs = Counter(artnames)
ordered_artnames = list(artfreqs.keys())
ordered_artvalues = np.asarray(list(artfreqs.values()))/np.sum(np.asarray(list(artfreqs.values())))

print(ordered_artvalues.shape)

chosen_subset_ids = []
idxs = np.arange(0,len(ordered_artnames))
for i in range(0,Nreps):
    chosen_subset_ids.append(np.random.choice(idxs,size=nartists,replace=False,p=np.squeeze(ordered_artvalues)))

for n in range(0,Nreps):

    patience_epochs = 0
    chosen_subset = []
    for i in range(0,nartists):
        chosen_subset.append(ordered_artnames[chosen_subset_ids[n][i]])

    test_ids_valid = []
    for ct,name in enumerate(artnames):
        if artnames[ct] in chosen_subset:
            test_ids_valid.append(ct)
        

    if vocal_mode:
        artist_id_data_ids = [test_ids_filt[x][:-4]+'_vocals.npy' for x in test_ids_valid]
    else:
        artist_id_data_ids = [test_ids_filt[x][:-4]+'.npy' for x in test_ids_valid]
  
    artist_id_labels_init = [artnames[x] for x in test_ids_valid]
    label_enc = OneHotEncoder(sparse_output=False).fit(np.reshape(np.asarray([artnames[x] for x in test_ids_valid]),(-1,1)))
    artist_id_labels = label_enc.transform(np.reshape(np.asarray([artnames[x] for x in test_ids_valid]),(-1,1)))

    sorted_index = [i for i, x in sorted(enumerate(artist_id_data_ids), key=lambda x: x[1]) ]
    ids_sorted = []
    for i in range(0,len(sorted_index)):
        ids_sorted.append(artist_id_data_ids[sorted_index[i]])    

    ids_unique = [ids_sorted[0]]
    labels_unique = np.zeros((len(ids_sorted),nartists))
    labels_unique[0,:] = artist_id_labels[sorted_index[0]]
    ctt = 1
    for i in range(1,len(ids_sorted)):
        if ids_sorted[i][:16] != ids_sorted[i-1][:16]:
            ids_unique.append(ids_sorted[i][:16])
            labels_unique[ctt,:] = artist_id_labels[sorted_index[i]]
            ctt+=1    
    
    labels_unique = labels_unique[:ctt]

    print(len(ids_unique),labels_unique.shape)
    indices = np.random.permutation((labels_unique.shape[0]))
    

    p1 = int(perc*0.8*len(indices))
    p2 = int(perc*0.9*len(indices))

    ids_unique_rad = [ids_unique[x] for x in indices]
    labels_unique_rad = labels_unique[indices,:]
    train_ids = ids_unique_rad[:p1]

    train_data = []
    train_labels = np.zeros((len(test_ids_filt),nartists))

    valid_ids_unique = ids_unique_rad[p1:p2]
    test_ids_unique = ids_unique_rad[int(.9*len(indices)):]
    valid_labels_unique = labels_unique_rad[p1:p2]
    test_labels_unique = labels_unique_rad[int(.9*len(indices)):]
    
    
    ctt = 0
    for i in range(0,len(test_ids_filt)):
        key = test_ids_filt[i][:16]
        if key in train_ids:
            idx = train_ids.index(key)
            if not vocal_mode:
                train_data.append(test_ids_filt[i])
            else:
                train_data.append(test_ids_filt[i][:-4]+'_vocals.npy')
            train_labels[ctt,:]  = labels_unique_rad[idx]
            ctt+=1

    train_labels = train_labels[:ctt]

    print(len(train_data))
    print(len(train_ids),len(valid_ids_unique),len(test_ids_unique))
    print(list(set(train_ids)&set(valid_ids_unique)))
    print(list(set(train_ids)&set(test_ids_unique)))

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
    x = encoder(x) #pretrained encoder of the CSSL model
    outputs = tf.keras.layers.Dense(nartists,activation='softmax')(x) #downstream model for tasks

    model = tf.keras.Model(inputs, outputs)
    model.get_layer("encoder").trainable = False #True for finetuning
    model.compile(
         optimizer=tf.keras.optimizers.Adam(0.0005),
          loss=tf.keras.losses.CategoricalCrossentropy(),
          metrics=[tf.keras.metrics.CategoricalAccuracy()])

    checkpoint = tf.train.Checkpoint(model)


    print(len(valid_ids_unique))
    best_val_acc = 0
    nEpochs = 100 
    batchSize = 8192
    for i in range (0,nEpochs):
    
        running_loss = 0
        running_acc = 0
    
        TrainPerm = np.random.permutation(len(train_data))
        train_samples = np.zeros((len(train_data),16000))
        train_labels_batch = np.zeros((len(train_data),nartists))

        for k in range(0,len(TrainPerm)):
            if vocal_mode:
                temp = np.load(datapath+'vocals/'+train_data[TrainPerm[k]])
            else:
                temp = np.load(datapath+'full/'+train_data[TrainPerm[k]])           
            sample_start = np.random.choice(64000)
            train_samples[k,:] = temp[sample_start:sample_start+16000]
            train_labels_batch[k,:] = train_labels[TrainPerm[k],:]
    
        evals = model.fit(x=train_samples,y=train_labels_batch,batch_size=64,epochs=1,verbose=2)
        running_loss += evals.history["loss"][0]
        running_acc += evals.history["categorical_accuracy"][0]

        gc.collect()
        print("Epoch", i, "train loss", running_loss, "train accuracy", running_acc)    
        valid_samples = np.zeros((len(valid_labels_unique)*30,16000))
        valid_labels_batch = np.zeros((len(valid_labels_unique)*30,nartists))
        numparts = np.zeros((len(valid_labels_unique)+1,),dtype=np.int32)
        ct = 0
        for k in range(0,len(valid_labels_unique)): #we do not sample those; instead we evaluate through the whole dataset
            for n in range(0,6):
                if vocal_mode:
                    if os.path.isfile(datapath+'vocals/'+valid_ids_unique[k]+'_down_'+str(n)+'_vocals.npy'):
                        temp = np.load(datapath+'vocals/'+valid_ids_unique[k]+'_down_'+str(n)+'_vocals.npy')
                        valid_samples[ct:ct+5,:] = np.reshape(temp,(5,16000))
                        valid_labels_batch[ct:ct+5,:] = np.tile(valid_labels_unique[k,:],(5,1))
                        ct += 5

                else:
                    if os.path.isfile(datapath+'vocals/'+valid_ids_unique[k]+'_down_'+str(n)+'_vocals.npy'):
                        temp = np.load(datapath+'full/'+valid_ids_unique[k]+'_down_'+str(n)+'.npy')        
                        valid_samples[ct:ct+5,:] = np.reshape(temp,(5,16000))
                        valid_labels_batch[ct:ct+5,:] = np.tile(valid_labels_unique[k,:],(5,1))
                        ct += 5
    
            numparts[k+1] = ct

        valid_samples = valid_samples[:ct,:]
        valid_labels_batch = valid_labels_batch[:ct,:]

        labels_inst = []
        labels_segs = []
        labels_segs2 = []
        labels_segs5 = []

        pred_labels_inst = []
        pred_labels_segs = []
        pred_labels_segs2 = []
        pred_labels_segs5 = []


        preds = model.predict(valid_samples)
        for k in range(0,len(valid_ids_unique)):
            if (numparts[k+1] - numparts[k]) > 0:
                    meanpreds = np.mean(preds[numparts[k]:numparts[k+1]],axis=0)
                    meanlabel = np.argmax(meanpreds)
                    pred_labels_inst.append(meanlabel)
                    labels_inst.append(np.argmax(valid_labels_unique[k,:]))

                    preds_local = preds[numparts[k]:numparts[k+1]]
                    labels_local = np.argmax(preds_local,axis=1)
                    for kk in range(0,len(labels_local)):
                        pred_labels_segs.append(labels_local[kk])
                        labels_segs.append(np.argmax(valid_labels_unique[k,:]))
                
                    temp = np.zeros((numparts[k+1]-numparts[k]+1,50))
                    for qq in range(0,50):
                        temp[:,qq] = np.convolve(preds_local[:,qq],[0.5,0.5],'full')
                    preds_subsampled = temp[1::2,:]
                    labels_local = np.argmax(preds_subsampled,axis=1)
                    for kk in range(0,len(labels_local)):
                        pred_labels_segs2.append(labels_local[kk])
                        labels_segs2.append(np.argmax(valid_labels_unique[k,:]))
                
                    temp = np.zeros((numparts[k+1]-numparts[k]+4,50))
                    for qq in range(0,50):
                        temp[:,qq] = np.convolve(preds_local[:,qq],[0.2,0.2,0.2,0.2,0.2],'full')
                    preds_subsampled = temp[4::5,:]
                    labels_local = np.argmax(preds_subsampled,axis=1)
                    for kk in range(0,len(labels_local)):   
                        pred_labels_segs5.append(labels_local[kk])
                        labels_segs5.append(np.argmax(valid_labels_unique[k,:]))
        
        segvalid = np.sum(np.asarray(pred_labels_segs) == np.asarray(labels_segs))/len(labels_segs) 
        segvalid2 = np.sum(np.asarray(pred_labels_segs2) == np.asarray(labels_segs2))/len(labels_segs2) 
        segvalid5 = np.sum(np.asarray(pred_labels_segs5) == np.asarray(labels_segs5))/len(labels_segs5) 
        fullvalid = np.sum(np.asarray(pred_labels_inst) == np.asarray(labels_inst))/len(labels_inst) 

        segvalid_f1 = f1_score(labels_segs,pred_labels_segs,average='macro')
        segvalid2_f1 = f1_score(labels_segs2,pred_labels_segs2,average='macro')
        segvalid5_f1 = f1_score(labels_segs5,pred_labels_segs5,average='macro')
        fullvalid_f1 = f1_score(labels_inst,pred_labels_inst,average='macro')
        
        print('segment-wise valid accuracy',segvalid) 
        print('2-segment-wise valid accuracy',segvalid2)  
        print('5-segment-wise valid accuracy',segvalid5)    
        print('valid accuracy',fullvalid)
    
        print('f1 scores',segvalid_f1, segvalid2_f1, segvalid5_f1,fullvalid_f1)
        if fullvalid > best_val_acc:
            best_val_acc = fullvalid
            patience_epochs = 0

            print('testing')
            test_samples = np.zeros((len(test_labels_unique)*30,16000))
            test_labels_batch = np.zeros((len(test_labels_unique)*30,nartists))
            numparts = np.zeros((len(test_labels_unique)+1,),dtype=np.int32)
            ct = 0
            for k in range(0,len(test_labels_unique)): #we do not sample those; instead we evaluate through the whole dataset
                for n in range(0,6):
                    if vocal_mode:
                        if os.path.isfile(datapath+'vocals/'+test_ids_unique[k]+'_down_'+str(n)+'_vocals.npy'):
                            temp = np.load(datapath+'vocals/'+test_ids_unique[k]+'_down_'+str(n)+'_vocals.npy')
                            test_samples[ct:ct+5,:] = np.reshape(temp,(5,16000))
                            test_labels_batch[ct:ct+5,:] = np.tile(test_labels_unique[k,:],(5,1))
                            ct += 5

                    else:
                        if os.path.isfile(datapath+'vocals/'+test_ids_unique[k]+'_down_'+str(n)+'_vocals.npy'):
                            temp = np.load(datapath+'full/'+test_ids_unique[k]+'_down_'+str(n)+'.npy')        
                            test_samples[ct:ct+5,:] = np.reshape(temp,(5,16000))
                            test_labels_batch[ct:ct+5,:] = np.tile(test_labels_unique[k,:],(5,1))
                            ct += 5
        
                numparts[k+1] = ct

            test_samples = test_samples[:ct,:]
            test_labels_batch = test_labels_batch[:ct,:]

            labels_inst = []
            labels_segs = []
            labels_segs2 = []
            labels_segs5 = []

            pred_labels_inst = []
            pred_labels_segs = []
            pred_labels_segs2 = []
            pred_labels_segs5 = []

            preds = model.predict(test_samples)
            for k in range(0,len(test_ids_unique)):
                if (numparts[k+1] - numparts[k]) > 0:
                    meanpreds = np.mean(preds[numparts[k]:numparts[k+1]],axis=0)
                    meanlabel = np.argmax(meanpreds)
                    pred_labels_inst.append(meanlabel)
                    labels_inst.append(np.argmax(test_labels_unique[k,:]))


                    preds_local = preds[numparts[k]:numparts[k+1]]
                    labels_local = np.argmax(preds_local,axis=1)
                    for kk in range(0,len(labels_local)):
                        pred_labels_segs.append(labels_local[kk])
                        labels_segs.append(np.argmax(test_labels_unique[k,:]))
                
                    temp = np.zeros((numparts[k+1]-numparts[k]+1,50))
                    for qq in range(0,50):
                        temp[:,qq] = np.convolve(preds_local[:,qq],[0.5,0.5],'full')
                    preds_subsampled = temp[1::2,:]
                    labels_local = np.argmax(preds_subsampled,axis=1)
                    for kk in range(0,len(labels_local)):
                        pred_labels_segs2.append(labels_local[kk])
                        labels_segs2.append(np.argmax(test_labels_unique[k,:]))
                
                    temp = np.zeros((numparts[k+1]-numparts[k]+4,50))
                    for qq in range(0,50):
                        temp[:,qq] = np.convolve(preds_local[:,qq],[0.2,0.2,0.2,0.2,0.2],'full')
                    preds_subsampled = temp[4::5,:]
                    labels_local = np.argmax(preds_subsampled,axis=1)
                    for kk in range(0,len(labels_local)):   
                        pred_labels_segs5.append(labels_local[kk])
                        labels_segs5.append(np.argmax(test_labels_unique[k,:]))

            segtest = np.sum(np.asarray(pred_labels_segs) == np.asarray(labels_segs))/len(labels_segs) 
            segtest2 = np.sum(np.asarray(pred_labels_segs2) == np.asarray(labels_segs2))/len(labels_segs2) 
            segtest5 = np.sum(np.asarray(pred_labels_segs5) == np.asarray(labels_segs5))/len(labels_segs5) 
            fulltest = np.sum(np.asarray(pred_labels_inst) == np.asarray(labels_inst))/len(labels_inst) 


            segtest_f1 = f1_score(labels_segs,pred_labels_segs,average='macro')
            segtest2_f1 = f1_score(labels_segs2,pred_labels_segs2,average='macro')
            segtest5_f1 = f1_score(labels_segs5,pred_labels_segs5,average='macro')
            fulltest_f1 = f1_score(labels_inst,pred_labels_inst,average='macro')
            
            print('segment-wise test accuracy',segtest) 
            print('2-segment-wise test accuracy',segtest2)  
            print('5-segment-wise test accuracy',segtest5)    
            print('test accuracy',fulltest)

            print('test macro-f1 scores', segtest_f1, segtest2_f1, segtest5_f1, fulltest_f1)
        else:
            patience_epochs += 1
            if patience_epochs > 5:
                metrics.append(segtest)
                metrics.append(segtest2)
                metrics.append(segtest5)
                metrics.append(fulltest)
                metrics.append(segtest_f1)
                metrics.append(segtest2_f1)
                metrics.append(segtest5_f1)
                metrics.append(fulltest_f1)

                break
        
        print('best val acc', best_val_acc)
       
        gc.collect()

metrics = np.reshape(np.asarray(metrics),(5,8))
print('over-experiment acc means',np.mean(metrics[:,:4],axis=0),'over-experiment acc stds',np.std(metrics[:,:4],axis=0))
print('over-experiment macrof1 means',np.mean(metrics[:,4:],axis=0),'over-experiment macrof1 stds',np.std(metrics[:,4:],axis=0))
