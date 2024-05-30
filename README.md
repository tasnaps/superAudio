Audio Enhancement Tool (WIP)
Overview

Idea here is to use U-net neural network similar to Eloi Moliner and Vesa Välimäki (http://research.spa.aalto.fi/publications/papers/icassp22-denoising/) to train a model to possibly restore high and low frequencies.
There is a possibility to train another model furhter to enhance the music recordings by expanding the dynamics - especially if we are dealing with very compressed material.


Currently what we have implemented:
U-net implementation
Conversion from audio to spectrograms.
Audio utilities to modify HQ audio samples.

Currently in progress:
Spectrogram to model training

TODO
Model performance evaluation.
Evaluation metrics and estimation on further training needs.
Validation set (prevent overfitting)
