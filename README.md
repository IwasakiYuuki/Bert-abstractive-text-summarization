# Abstractive text summarization using BERT
This is the models using BERT (refer the paper [Pretraining-Based Natural Language Generation for Text Summarization
](https://arxiv.org/abs/1902.09243) ) for one of the NLP(Natural Language Processing) task, abstractive text summarization.
 
## Requirements
- Python 3.6.5+
- Pytorch 0.4.1+
- tqdm
- Numpy
- Juman++
- Tensorboard X and others...

All packages used here can be installed by pip as follow:

~~~
pip install -r requirement.txt
~~~

please refer [here](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB) when installing Juman++.

## Docker
If you train the model with GPU, it is easy to use [Pytorch docker images](https://hub.docker.com/r/pytorch/pytorch) in DockerHub.
 
In this study, pytorch/pytorch:0.4.1-cuda9-cudnn7-devel(2.62GB) has been used.

## Before using
When you use this, please follow the steps below. 
1. Make a repository named "/data/checkpoint" under root. 
And put bert_model, vocabulary file and config file for bert. 
These files can be download [here](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB).

2. Put data file for training and validate under /data/. The format is as follow:

```preprocess.py
data = {
    'settings': opt,
    'dict': {
        'src': text2token,
        'tgt': text2token},
    'train': {
        'src': content[:100000],
        'tgt': summary[:100000]},
    'valid': {
        'src': content[100000:],
        'tgt': summary[100000:]}}
torch.save(data, opt.save_data)
```

overall directory structure is as follow:
```
`-- data                        # under root 
    |-- checkpoint
    |   |-- bert_config.json    # BERT config file
    |   |-- pytorch_model.bin   # BERT model file
    |   `-- vocab.txt           # vocabulary file
    `-- preprocessed_data.data  # train and valid data file
```

## Usage
### Train the model
```
python train.py -data data/preprocessed_data.data -bert_path data/checkpoint/ -proj_share_weight -label_smoothing -batch_size 4 -epoch 10 -save_model trained -save_mode best
```

## Future works
- Eval the model with score such as ROUGE-N
- Upload train log graph using TensorboardX
- Make some examples