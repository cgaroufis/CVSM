# CVSM: Contrastive Vocal Similarity Modeling

Companion repository for the (under review) paper CVSM: Contrastive Vocal Similarity Modeling, available on [arxiv](https://arxiv.org/pdf/2510.03025).

# Preliminaries

To execute the code hosted in the current git repo, you need a ```python 3.10``` environment. The models were trained on ```tensorflow-gpu 2.8.1```, while ```librosa 0.9.2``` was used for audio excerpt pre-processing. For obtaining vocal source excerpts, an off-the-shelf source separation model is also required; in our implementation, we used [open-unmix](https://github.com/sigsep/open-unmix-pytorch).

You can adapt the given code to any musical audio dataset with artist metadata, but for off-the-shelf pre-training, and to be able to reproduce the paper results, use the Music4All dataset; it is avaiable [upon request](https://sites.google.com/view/contact4music4all).

# Data pre-processing

The audio files in Music4All are provided in ```.mp3``` format; run the ```./get_wavs.sh``` script to transform them to ```.wav``` files, followed by the ```./downsample_wavs.sh``` script to downsample them to 16 kHz.

To obtain the vocal stems, run the ```./get_stems.sh``` script on the transformed ```.wav``` files; the script also includes downsampling at 16kHz.

# Model pre-training

To pre-train a CVSM model, you can execute:

```python3 pretraining.py data_dir model_dir --configs myconfig.ini```

You can structure your config file similar to those contained in the ```configs``` subfolder, which contain the following properties:

```augment```: Whether artificial vocal-accompaniment mixtures are created as anchor instances.

```artist```: Whether sampling occurs at artist level

```sources```: Is either ```None``` (for mixture-mixture pre-training) or ```vocals``` (for mixture-vocal pre-training)

```finetune```: Path to the model to fine-tune/continue training; if trained from scratch, set to ```None```.

# Model evaluation

We provide scripts for evaluating the pre-trained models in the following downstream tasks:

* Gender Identification (```gender_id_testing.py```)
  
* Artist Identification (```artist_id_testing.py```)
  
* Artist Similarity (```artist_sim_testing.py```)
