# AT-DCNet
This repository contains the codebase for AT-DCNet, 
implemented using PyTorch for conducting experiments on LEVIR-CD and CDD. 
## Data preparation:
```
train 
  --A 
  --B
  --label
val 
  --A 
  --B
  --label  
test  
  --A 
  --B
  --label  
```
## FIGS        

![image](/figs/fig1.jpg)
## Datasets

LEVIR-CD:
[https://justchenhao.github.io/LEVIR/]  OR [https://opendatalab.org.cn/OpenDataLab/LEVIR-CD]

CDD:
[https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9]

## Environments

1.CUDA  11.8

2.Pytorch 2.1

3.Python 3.11

## Train and Test
The training and evaling pipeline is organized in train_cd.py and eval_cd.py.
```bash
python train_cd.py
```

```bash
python eval_cd.py
```