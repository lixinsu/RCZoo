
## Introduction
The RCZoo project is a toolkit for reading comprehension model. It contains the [PyTorch](https://pytorch.org/) reimplement of multiple reading comprehension model.  
All the models are trained and tested on the SQuAD v1.1 dataset, and reach the performance in origin papers.  

## Dependencies
python 3.5  
Pytorch 0.4  
tqdm  


## performance
We train each model on train set for 40 epoch, and report the best performance on dev set.  

Model | Exact Match | F1  
---- | --- | ---  
Rnet | 69.25 | 78.97 
BiDAF | 70.47 | 79.90 
documentqa | 71.47 | 80.84 
DrQA | 68.39 | 77.90 
QAnet | ... | ... 
SLQA | ... | ... 
FusionNet | ... | ... 


## Current progress
### [Rnet](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
- [x] training
- [x] performance
- [ ] predicting scripts  
`some different in the Dropout Layer`
### [BiDAF](https://arxiv.org/abs/1611.01603)
- [x] training
- [x] performance
- [ ] predicting scripts  
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
### [FusionNet](https://openreview.net/forum?id=BJIgi_eCZ&noteId=BJIgi_eCZ)
- [ ] training
- [ ] performance
- [ ] predicting scripts

## Usage
 - run `sh download.sh` to download the dataset and the glove embeddings. 
 - run `sh train_xxx.sh` to start the train process. Dring the train process, model will be evaluated on dev set each epoch.
 
 ## acknowledgement
  some code are borrowed from [DrQA](https://github.com/facebookresearch/DrQA.git), a cool project about reading comprehension.  
 
TODO:  
 - Recognizing unanswerable question for SQuAD, add new type of loss function to accommodate unanswerable question  
 - Processing multiple passage reading comprehension. Related datasets include [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [SearchQA](https://arxiv.org/abs/1704.05179), [QuasarT](https://arxiv.org/abs/1707.03904)

