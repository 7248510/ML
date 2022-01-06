#Source code location https://github.com/pytorch/text
#conda install -c pytorch torchtext
#https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
#Work in progress
#Sample configuration for text classification:
import torch
from torchtext.datasets import AG_NEWS
train_iter = AG_NEWS(split='train')
print(next(train_iter))
