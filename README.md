# Project Code for CSE6240 Course, Spring 2020
> Zan Huang, Irfan AI-Al-Hussaini, Weiyang Liu

We implmented three models for MOOC dropout prediction. [JODIE](https://github.com/srijankr/jodie) and RDP(our proposed dropout predictor) implementations are provided locally, [CFIN](https://github.com/wzfhaha/dropout_prediction) implementation was hosted on [Google Colab](https://drive.google.com/drive/folders/1Zsp7W17aqz_leWFt61_pWQo67QrZOdVg?usp=sharing) due to its using too large dataset.

For datasets, the smaller one would be automatically downloaded when you execuate code, or you can manually download it by [this link](http://snap.stanford.edu/jodie/mooc.csv)， the larger one could be downloaded from [moocdata.cn](http://moocdata.cn/data/user-activity) or directly accessed on [this Google Drive Folder](https://drive.google.com/drive/folders/1lNr4JXUjpevqm-9J34v5VVsy6wr3_iLa?usp=sharing) along with ipython notebook for CFIN implementation.

### Setup

```bash
pip install requirements.txt
```

other setup steps are coded in python scripts.

if you want to access pretrained models, pls visit [this google drive folder](https://drive.google.com/drive/folders/1-7hsvF6AP08R8wa947AN6KPU_dkCf19y?usp=sharing).

`logs` folder containing the output files during model training and inference.

### JODIE Experiment

```bash
python jodie.py
```

### RDP Experiment
```bash
python rdp.py
```

### CFIN Experiment

Pls visit [this Colab Notebook](https://colab.research.google.com/drive/1VfhzQ6FqaeIfwHoGuK1as2CQukc6ApE_) for accessing code and viewing experiment results.
