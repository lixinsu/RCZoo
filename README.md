
## Introduction
The RCZoo project is a toolkit for reading comprehension model. It contains the [PyTorch](https://pytorch.org/) reimplement of multiple reading comprehension models  

## Usage
 - run `sh download.sh` to download the dataset and the glove embeddings. 
 - run `sh runs/train_squad.sh [bidaf|drqa|slqa|fusionnet|docqa]` to start the train process. (Check the xxx.sh scripts before run, as the preprocessing only need to be executed once)

 
## Dependencies
python 3.5  
Pytorch 0.4  
tqdm  


## performance

We train each model on train set for 40 epoch, and report the best performance on dev set.  

Model | Exact Match | F1 | EM(+ELMo) | F1(+ELMo)
---- | --- | --- | --- | --- 
Rnet | 69.25 | 78.97 |
BiDAF | 70.47 | 79.90 | 73.04 | 81.48
documentqa | 71.47 | 80.84 | 
DrQA | 68.39 | 77.90 |
QAnet | ... | ... |
SLQA | 67.09 | 76.67 | 
FusionNet | 68.27 | 77.79 |  

## Current progress
### [Rnet](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
### [BiDAF](https://arxiv.org/abs/1611.01603)
### [documentqa](https://arxiv.org/abs/1710.10723)
### [DrQA](https://arxiv.org/abs/1704.00051)
### [QAnet](https://arxiv.org/abs/1804.09541)
### [SLQA](http://aclweb.org/anthology/P18-1158)
### [FusionNet](https://openreview.net/forum?id=BJIgi_eCZ&noteId=BJIgi_eCZ)

 ## acknowledgement
 some code are borrowed from [DrQA](https://github.com/facebookresearch/DrQA.git), a cool project about reading comprehension.  
 
