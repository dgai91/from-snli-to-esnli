# from-snli-to-esnli
  
These days, I'm trying to reproduct SNLI and e-SNLI models as an exercise. In this project, I will accomplish the following two tasks.

First, I hope to be able to complete all the experiments in the [original paper](https://arxiv.org/abs/1705.02364). 

Then, based on the InferSent model, I will implement [e-SNLI model](https://arxiv.org/abs/1812.01193v1) and complete corresponding experiments.

If you want to use my project in your repositories, all requirments as follow:

    Python 3.7 with anaconda

    Pytorch 1.2.0 GPU

    nltk with punkt model

    snli dataset and glove 840b 300d

### current progress:

All infersent models are complete, you can train these models using train.py (The first time you use this project, you need to run data_process.py to generate a small glove file and dataset)
