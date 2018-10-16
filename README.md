
## Introduction
The RCZoo project is a toolkit for reading comprehension model. It contains the [PyTorch](https://pytorch.org/) reimplement of multiple reading comprehension model.  
All the models are trained and tested on the SQuAD v1.1 dataset, and reach the performance in origin papers.  

## Usage
 - run `sh download.sh` to download the dataset and the glove embeddings. 

 - run `sh runs/train_squad.sh [bidaf|drqa|slqa|fusionnet|docqa]` to start the train process. (Check the xxx.sh scripts before run, as the preprocessing only need to be executed once)

 
## Dependencies
python 3.5  
Pytorch 0.4  
tqdm  


## performance

`new: We replace the orgin GloVE vector with ELMo embeddings. It bring a performance gain about 3 points in EM. The code is now in feature/elmo branch`  
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
- [x] training
- [x] performance
- [ ] predicting scripts  
`some different in the Dropout Layer`
### [BiDAF](https://arxiv.org/abs/1611.01603)
- [x] training
- [x] performance
- [x] predicting scripts  
- [x] +ELMo
`The bi-attention in BiDAF does not work fin, and I introduce the co-attention in DCN paper. The final results is better than that in origin paper`
### [documentqa](https://arxiv.org/abs/1710.10723)
- [x] training
- [x] performance
- [ ] predicting scripts
### [DrQA](https://arxiv.org/abs/1704.00051)
`borrow from origin code`
- [x] training
- [x] performance
- [ ] predicting scripts
### [QAnet](https://arxiv.org/abs/1804.09541)
- [x] training
- [ ] performance
- [ ] predicting scripts
### [SLQA](http://aclweb.org/anthology/P18-1158)
- [x] training
- [ ] performance
- [ ] predicting scripts   
`no elmo contextualized embedding`
### [FusionNet](https://openreview.net/forum?id=BJIgi_eCZ&noteId=BJIgi_eCZ)
- [x] training
- [ ] performance
- [ ] predicting scripts   
`no CoVe embedding`

 ## acknowledgement
  some code are borrowed from [DrQA](https://github.com/facebookresearch/DrQA.git), a cool project about reading comprehension.  
 
TODO:  
 - Recognizing unanswerable question for SQuAD, add new type of loss function to accommodate unanswerable question  
 - Processing multiple passage reading comprehension. Related datasets include [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [SearchQA](https://arxiv.org/abs/1704.05179), [QuasarT](https://arxiv.org/abs/1707.03904)

