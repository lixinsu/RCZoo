
## Introduction
The RCZoo project is a toolkit for reading comprehension model. It contains the [PyTorch](https://pytorch.org/) reimplement of multiple reading comprehension model.  
All the models are trained and tested on the SQuAD v1.1 dataset, and reach the performance in origin papers.  

## performance
We train each model on train set for 40 epoch, report the best performance on dev set.  

Model | Exact Match | F1  
---- | --- | ---  
Rnet | xx | xx 
BiDAF | xx | xx 
documentqa | xx | xx 
DrQA | xx | xx 
QAnet | xx | xx 


## Current progress
### [Rnet](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
- [x] training
- [x] performance
- [ ] predicting scripts
### [BiDAF](https://arxiv.org/abs/1611.01603)
- [x] training
- [x] performance
- [ ] predicting scripts
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


## Usage
 - run `sh download.sh` to download the dataset and the glove embeddings. 
 - run `sh train_xxx.sh` to start the train process. Dring the train process, model will be evaluated on dev set each epoch.
 
 ## acknowledgement
  some code are borrowed from [DrQA](https://github.com/facebookresearch/DrQA.git), a cool project about reading comprehension.  
 
TODO:  
 - Recognizing unanswerable question for SQuAD, add new type of loss function to accommodate unanswerable question  
 - Processing multiple passage reading comprehension. Related datasets include [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [SearchQA](https://arxiv.org/abs/1704.05179), [QuasarT](https://arxiv.org/abs/1707.03904)

