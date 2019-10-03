# Abstractive text summarization using BERT
 This is the models using BERT (refer the paper [Pretraining-Based Natural Language Generation for Text Summarization
](https://arxiv.org/abs/1902.09243) ) for one of the NLP(Natural Language Processing) task, abstractive text summarization.
 
## Requirement
- Python 3.6.5+
- Pytorch 0.4.1+
- tqdm
- Numpy
- Tensorboard X and others...

All packages used here can be installed by pip as follow:

~~~
pip install -r requirement.txt
~~~

## Docker
 If you train the model with GPU, it is easy to use [Pytorch docker images](https://hub.docker.com/r/pytorch/pytorch) in DockerHub.
 
 In this study, pytorch/pytorch:0.4.1-cuda9-cudnn7-devel(2.62GB) has been used.
