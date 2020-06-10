# Tweet Sentiment Extraction
[竞赛链接](https://www.kaggle.com/c/tweet-sentiment-extraction)
## 数据下载
[data](https://www.kaggle.com/c/tweet-sentiment-extraction/data)
## 评估标准
[Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index)
```python
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```
## kaggle score:
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert-base-uncased|0.6960|-1|b=45;0.6921,0.6983,0.6981,0.6987,0.6927|
|roberta-base|0.7096|-1|b=64;0.7111,0.7096,0.7127,0.7136,0.7009|
|roberta-large|-1|-1||

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert-base-uncased|70.34|-1|b=45;epoch=4就不再提升了|
|roberta-base+dist_loss_fn|71.18|0.708|b=64;epoch=8就不再提升了|
|roberta-large|0.707|0.695|b=8;epoch=2就不再提升了|
|roberta-large+loss_fn_plus|66.57|-1|b=8;epoch=2就不再提升了|
|roberta-large+dist_loss_fn|0.7067|-1|b=8;epoch=2就不再提升了|


## script
nohup runipy -o code.ipynb > info.out 2>&1 &  
nohup python main.py -o=train -m=bert -b=32 -e=4 -mode=2 > nohup/bert.out 2>&1 & 
nohup python main.py -o=train -m=roberta -b=16 -e=16 -mode=2 > nohup/roberta.out 2>&1 &  

nohup python main.py -o=train -m=roberta -b=64 -e=16 > nohup/roberta.out 2>&1 &  
nohup python main.py -o=train -m=roberta -b=16 -e=8 > nohup/roberta.out 2>&1 &  