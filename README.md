<div align="center"> <h1>Parkinson's Disease Detection</h1> </div>
<div align="center"><b>A modern machine learning project for detection of parkinson's disease</b></div>

- [Introduction](#introduction)
- [Packages](#packages)
- [Methodology](#methodology)
  - [Dataset Description](#dataset-description)
  - [Data Pre-Processing](#dataset-description)
  - [Applied Models](#applied-models)
- [Results](#results)
  - [Accuracy](#results)
  - [Visualisation 1](#visualization-1)
  - [Visualisation 2](#visualization-2)
- [Contributors](#contributors)
- [References](#references)

# Introduction

Parkinson's disease decription, reason for study (https://www.sciencedirect.com/science/article/abs/pii/S1386505618303344?via%3Dihub). Therefore, the pupose of this project is to help detecting it.


```python
import pandas as pd
import numpy as np

pd.options.display.notebook_repr_html = False
```


```python
# Data From https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection?resource=download
data = pd.read_csv('./Parkinsson disease.csv')
data
```




                   name  MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)  \
    0    phon_R01_S01_1      119.992       157.302        74.997         0.00784   
    1    phon_R01_S01_2      122.400       148.650       113.819         0.00968   
    2    phon_R01_S01_3      116.682       131.111       111.555         0.01050   
    3    phon_R01_S01_4      116.676       137.871       111.366         0.00997   
    4    phon_R01_S01_5      116.014       141.781       110.655         0.01284   
    ..              ...          ...           ...           ...             ...   
    190  phon_R01_S50_2      174.188       230.978        94.261         0.00459   
    191  phon_R01_S50_3      209.516       253.017        89.488         0.00564   
    192  phon_R01_S50_4      174.688       240.005        74.287         0.01360   
    193  phon_R01_S50_5      198.764       396.961        74.904         0.00740   
    194  phon_R01_S50_6      214.289       260.277        77.973         0.00567   
    
         MDVP:Jitter(Abs)  MDVP:RAP  MDVP:PPQ  Jitter:DDP  MDVP:Shimmer  ...  \
    0             0.00007   0.00370   0.00554     0.01109       0.04374  ...   
    1             0.00008   0.00465   0.00696     0.01394       0.06134  ...   
    2             0.00009   0.00544   0.00781     0.01633       0.05233  ...   
    3             0.00009   0.00502   0.00698     0.01505       0.05492  ...   
    4             0.00011   0.00655   0.00908     0.01966       0.06425  ...   
    ..                ...       ...       ...         ...           ...  ...   
    190           0.00003   0.00263   0.00259     0.00790       0.04087  ...   
    191           0.00003   0.00331   0.00292     0.00994       0.02751  ...   
    192           0.00008   0.00624   0.00564     0.01873       0.02308  ...   
    193           0.00004   0.00370   0.00390     0.01109       0.02296  ...   
    194           0.00003   0.00295   0.00317     0.00885       0.01884  ...   
    
         Shimmer:DDA      NHR     HNR  status      RPDE       DFA   spread1  \
    0        0.06545  0.02211  21.033       1  0.414783  0.815285 -4.813031   
    1        0.09403  0.01929  19.085       1  0.458359  0.819521 -4.075192   
    2        0.08270  0.01309  20.651       1  0.429895  0.825288 -4.443179   
    3        0.08771  0.01353  20.644       1  0.434969  0.819235 -4.117501   
    4        0.10470  0.01767  19.649       1  0.417356  0.823484 -3.747787   
    ..           ...      ...     ...     ...       ...       ...       ...   
    190      0.07008  0.02764  19.517       0  0.448439  0.657899 -6.538586   
    191      0.04812  0.01810  19.147       0  0.431674  0.683244 -6.195325   
    192      0.03804  0.10715  17.883       0  0.407567  0.655683 -6.787197   
    193      0.03794  0.07223  19.020       0  0.451221  0.643956 -6.744577   
    194      0.03078  0.04398  21.209       0  0.462803  0.664357 -5.724056   
    
          spread2        D2       PPE  
    0    0.266482  2.301442  0.284654  
    1    0.335590  2.486855  0.368674  
    2    0.311173  2.342259  0.332634  
    3    0.334147  2.405554  0.368975  
    4    0.234513  2.332180  0.410335  
    ..        ...       ...       ...  
    190  0.121952  2.657476  0.133050  
    191  0.129303  2.784312  0.168895  
    192  0.158453  2.679772  0.131728  
    193  0.207454  2.138608  0.123306  
    194  0.190667  2.555477  0.148569  
    
    [195 rows x 24 columns]


