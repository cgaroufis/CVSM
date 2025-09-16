# Evaluates the pre-trained contrastive encoder im the task of gender identification
# Usage: python3 gender_id_testing.py datapath model_dir/ perc [--vocal_mode]
# datapath: data storage directory
# model_dir: directory of the pre-trained model
# --vocal_mode: evaluates on isolated vocals; if not included, evaluates on complete mixtures.


import gc
import os
import sys
import argparse
import numpy as np
import pandas as pd
from cola import constants
from mscol import network
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
metrics = []

for kk in range(0,10):

    train_ids_filt = np.load('splits/gender_train_ids_'+str(kk)+'.npy',allow_pickle=True)
    valid_ids_filt = np.load('splits/gender_valid_ids_'+str(kk)+'.npy',allow_pickle=True)
    test_ids_filt = np.load('splits/gender_test_ids_'+str(kk)+'.npy',allow_pickle=True)

    train_labels = np.load('splits/gender_train_labels_'+str(kk)+'.npy')
    valid_labels = np.load('splits/gender_valid_labels_'+str(kk)+'.npy')
    test_labels = np.load('splits/gender_test_labels_'+str(kk)+'.npy')

    print(len(train_ids_filt))

    # Load contrastive network from pretrained dir

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
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x) #downstream model for tasks

    model = tf.keras.Model(inputs, outputs)
    model.get_layer("encoder").trainable = False #True for finetuning

    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()])

    checkpoint = tf.train.Checkpoint(model)

    train_labels = np.asarray(train_labels)
    valid_labels = np.asarray(valid_labels)
    test_labels = np.asarray(test_labels)

    sort_index_valids = [i for i, x in sorted(enumerate(valid_ids_filt), key=lambda x: x[1])]
    valid_ids_sorted = []
    for i in range(0,len(sort_index_valids)):
        valid_ids_sorted.append(valid_ids_filt[sort_index_valids[i]])

    valid_ids_unique = [valid_ids_sorted[0]]
    valid_labels_unique = [valid_labels[sort_index_valids[0]]]
    for i in range(1,len(valid_ids_filt)):
        if valid_ids_sorted[i][:16] != valid_ids_sorted[i-1][:16]:
            valid_ids_unique.append(valid_ids_sorted[i])
            valid_labels_unique.append(valid_labels[sort_index_valids[i]])

    print(len(valid_ids_unique))

    sort_index_tests = [i for i, x in sorted(enumerate(test_ids_filt), key=lambda x: x[1])]
    test_ids_sorted = []
    for i in range(0,len(sort_index_tests)):
        test_ids_sorted.append(test_ids_filt[sort_index_tests[i]])

    test_ids_unique = [test_ids_sorted[0]]
    test_labels_unique = [test_labels[sort_index_tests[0]]]
    for i in range(1,len(test_ids_filt)):
        if test_ids_sorted[i][:16] != test_ids_sorted[i-1][:16]:
            test_ids_unique.append(test_ids_sorted[i])
            test_labels_unique.append(test_labels[sort_index_tests[i]])

    print(len(test_ids_unique))

    best_val_acc = 0
    nEpochs = 50 
    batchSize = 8192
    patience = 0
    for i in range (0,nEpochs):
        
        TrainPerm = np.random.permutation(len(train_ids_filt))[:batchSize]
        train_samples = np.zeros((batchSize,16000))
        train_labels_batch = np.zeros((batchSize,))

        for k in range(0,batchSize):
            if vocal_mode:
                temp = np.load(datapath+'/vocals/'+train_ids_filt[TrainPerm[k]][:-4]+'_vocals.npy')
            else:
                temp = np.load(datapath+'/full/'+train_ids_filt[TrainPerm[k]][:-4]+'.npy')

            sample_start = np.random.choice(64000)
            train_samples[k,:] = temp[sample_start:sample_start+16000]
            train_labels_batch[k] = train_labels[TrainPerm[k]]
    
        evals = model.fit(x=train_samples,y=train_labels_batch,batch_size=64,epochs=1,verbose=2) #TODO: return losses/accs so as to average
        running_loss = evals.history["loss"][0]
        running_acc = evals.history["binary_accuracy"][0]

        gc.collect()
        print("Epoch", i, "train loss", running_loss, "train accuracy", running_acc)    
        valid_samples = np.zeros((len(valid_labels_unique)*30,16000))
        valid_labels_batch = np.zeros((len(valid_labels_unique)*30,))
        numparts = np.zeros((len(valid_labels_unique)+1,),dtype=np.int32)
        ct = 0


        for k in range(0,len(valid_labels_unique)): #we do not sample those; instead we evaluate through the whole dataset
            for n in range(0,6):
                if os.path.isfile(datapath+'/vocals/'+valid_ids_unique[k][:-11]+'_down_'+str(n)+'_vocals.npy'):
                    if vocal_mode:
                        temp = np.load(datapath+'/vocals/'+valid_ids_unique[k][:-11]+'_down_'+str(n)+'_vocals.npy')
                    else:
                        temp = np.load(datapath+'/full/'+valid_ids_unique[k][:-11]+'_down_'+str(n)+'.npy')

                    valid_samples[ct:ct+5,:] = np.reshape(temp,(5,16000)) 
                    valid_labels_batch[ct:ct+5] = np.tile(valid_labels_unique[k],(5,))
                    ct += 5
            numparts[k+1] = ct

        valid_samples = valid_samples[:ct,:]
        valid_labels_batch = valid_labels_batch[:ct]

        corrinst = 0
        inst = 0

        corrinst_segs = 0
        inst_segs = 0

        corrinst_segs2 = 0
        inst_segs2 = 0

        corrinst_segs5 = 0
        inst_segs5 = 0

        preds = model.predict(valid_samples)
        for k in range(0,len(valid_ids_unique)):
            if (numparts[k+1] - numparts[k]) > 0:
                meanlabel = np.mean(preds[numparts[k]:numparts[k+1]])
                if (meanlabel>0.5)==valid_labels_unique[k]:
                    corrinst += 1
                inst += 1

            labels_local = preds[numparts[k]:numparts[k+1]]
            corrinst_segs += np.sum(np.round(labels_local) == valid_labels_unique[k])
            inst_segs += len(labels_local)  
        
            temp = np.convolve(np.squeeze(labels_local),[0.5,0.5],'full')
            labels_subsampled = temp[1::2]
            corrinst_segs2 += np.sum(np.round(labels_subsampled) == valid_labels_unique[k])
            inst_segs2 += len(labels_subsampled)  


            temp = np.convolve(np.squeeze(labels_local),[0.2,0.2,0.2,0.2,0.2],'full')
            labels_subsampled = temp[4::5]
            corrinst_segs5 += np.sum(np.round(labels_subsampled) == valid_labels_unique[k])
            inst_segs5 += len(labels_subsampled)  
        
        print('segment-wise validation accuracy',corrinst_segs/inst_segs)
        print(corrinst_segs,inst_segs)   
        print('2-segment-wise validation accuracy',corrinst_segs2/inst_segs2)
        print(corrinst_segs2,inst_segs2)    
        print('5-segment-wise validation accuracy',corrinst_segs5/inst_segs5)
        print(corrinst_segs5,inst_segs5)     
        print('validation accuracy',corrinst/inst)
        print(corrinst, inst)

        gc.collect()
        if corrinst/inst > best_val_acc:
            best_val_acc = corrinst/inst

            print('testing..')
            test_samples = np.zeros((len(test_labels_unique)*30,16000))
            test_labels_batch = np.zeros((len(test_labels_unique)*30,))
            numparts = np.zeros((len(test_labels_unique)+1,),dtype=np.int32)
            ct = 0

            for k in range(0,len(test_labels_unique)): #we do not sample those; instead we evaluate through the whole dataset
                for n in range(0,6):
                    if os.path.isfile(datapath+'/vocals/'+test_ids_unique[k][:-11]+'_down_'+str(n)+'_vocals.npy'):
                        if vocal_mode:
                            temp = np.load(datapath+'/vocals/'+test_ids_unique[k][:-11]+'_down_'+str(n)+'_vocals.npy')
                        else:    
                            temp = np.load(datapath+'/full/'+test_ids_unique[k][:-11]+'_down_'+str(n)+'.npy')

                        test_samples[ct:ct+5,:] = np.reshape(temp,(5,16000)) 
                        test_labels_batch[ct:ct+5] = np.tile(test_labels_unique[k],(5,))
                        ct += 5

                numparts[k+1] = ct

            test_samples = test_samples[:ct,:]
            test_labels_batch = test_labels_batch[:ct]

            corrinst = 0
            inst = 0

            corrinst_segs = 0
            inst_segs = 0

            corrinst_segs2 = 0
            inst_segs2 = 0

            corrinst_segs5 = 0
            inst_segs5 = 0

            preds = model.predict(test_samples)
            for k in range(0,len(test_ids_unique)):
                if (numparts[k+1] - numparts[k]) > 0:
                    meanlabel = np.mean(preds[numparts[k]:numparts[k+1]])
                    if (meanlabel>0.5)==test_labels_unique[k]:
                        corrinst += 1
                    inst += 1

                labels_local = preds[numparts[k]:numparts[k+1]]
                corrinst_segs += np.sum(np.round(labels_local) == test_labels_unique[k])
                inst_segs += len(labels_local)  
            
                temp = np.convolve(np.squeeze(labels_local),[0.5,0.5],'full')
                labels_subsampled = temp[1::2]
                corrinst_segs2 += np.sum(np.round(labels_subsampled) == test_labels_unique[k])
                inst_segs2 += len(labels_subsampled)  


                temp = np.convolve(np.squeeze(labels_local),[0.2,0.2,0.2,0.2,0.2],'full')
                labels_subsampled = temp[4::5]
                corrinst_segs5 += np.sum(np.round(labels_subsampled) == test_labels_unique[k])
                inst_segs5 += len(labels_subsampled)  

            test_seg_acc  =corrinst_segs/inst_segs
            print('segment-wise testing accuracy',corrinst_segs/inst_segs)
            test_seg_2acc = corrinst_segs2/inst_segs2 
            print('2-segment-wise testing accuracy',corrinst_segs2/inst_segs2)
            test_seg_5acc = corrinst_segs5/inst_segs5  
            print('5-segment-wise testing accuracy',corrinst_segs5/inst_segs5)
            test_acc = corrinst/inst
            print('testing accuracy',corrinst/inst)
               
            patience = 0
        else:
            patience+=1
            if patience > 5:
                metrics += [test_seg_acc,test_seg_2acc,test_seg_5acc,test_acc]
                break
        print('best val acc', best_val_acc)
        gc.collect()


_metrics = np.asarray(metrics).reshape((10,4)).mean(axis=0)
print('experiment metrics (acc%)', _metrics)
        


