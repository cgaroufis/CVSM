# CVSM: Contrastive Vocal Similarity Modeling

Companion repository for the (under review) paper CVSM: Contrastive Vocal Similarity Modeling, available on [arxiv](https://arxiv.org/pdf/2510.03025).

# Preliminaries

To execute the code hosted in the current git repo, you need a ```python 3.10``` environment. The models were trained on ```tensorflow-gpu 2.8.1```, while ```librosa 0.9.2``` was used for audio excerpt pre-processing. For obtaining vocal source excerpts, you also need an off-the-shelf source separation model; in our implementation, we used open-unmix.

You can adapt the given code to any musical audio dataset with artist metadata, but for off-the-shelf pre-training, and to be able to reproduce the paper results, use the Music4All dataset; it is avaiable upon request.

# Model pre-training

To pre-train a CVSM model, you can execute:

```python3 pretraining.py```

# Model evaluation

We provide scripts for evaluating the pre-trained models in the following downstream tasks:

* Gender Identification ()
* Artist Identification ()
* Artist Similarity ()
