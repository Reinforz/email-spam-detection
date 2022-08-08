<div align="center"> <h1>Parkinson's Disease Detection</h1> </div>
<div align="center"><b>A modern machine learning project for detection of parkinson's disease</b></div>

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Dataset Description](#dataset-description)
  - [Data Pre-Processing](#data-pre-processing)
  - [Applied Models](#applied-models)
- [Results](#results)
  - [Accuracy](#accuracy)
  - [Visualisation 1](#visualization-1)
  - [Visualisation 2](#visualization-2)
- [Contributors](#contributors)
- [References](#references)

# Introduction

Parkinson's disease decription, reason for study [[1]](#cite1). Therefore, the pupose of this project is to help detecting it.

# Methodology
TO-DO soon


```python
import pandas as pd
import numpy as np

from random import randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
```

### Dataset Description
Data taken from [[2]](#cite2)

##### Attribute Information:
- name - subject name and recording number
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency
- MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP - Several measures of variation in fundamental frequency
- MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA - Several measures of variation in amplitude
- NHR, HNR - measures of ratio of noise to tonal components in the voice
- status - one means subject has Parkinson's, zero means subject is healthy
- RPDE, D2 - nonlinear dynamical complexity measures
- DFA - Signal fractal scaling exponent
- spread1, spread2, PPE - Three nonlinear measures of fundamental frequency variation


```python
# Data From https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection?resource=download
fullData = pd.read_csv('./Parkinsson disease.csv')
num = len(fullData.name)
fullData.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>MDVP:Shimmer(dB)</th>
      <th>...</th>
      <th>MDVP:APQ</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>119.992</td>
      <td>157.302</td>
      <td>74.997</td>
      <td>0.00784</td>
      <td>0.00007</td>
      <td>0.00370</td>
      <td>0.00554</td>
      <td>0.01109</td>
      <td>0.04374</td>
      <td>0.426</td>
      <td>...</td>
      <td>0.02971</td>
      <td>0.06545</td>
      <td>0.02211</td>
      <td>21.033</td>
      <td>0.414783</td>
      <td>0.815285</td>
      <td>-4.813031</td>
      <td>0.266482</td>
      <td>2.301442</td>
      <td>0.284654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>122.400</td>
      <td>148.650</td>
      <td>113.819</td>
      <td>0.00968</td>
      <td>0.00008</td>
      <td>0.00465</td>
      <td>0.00696</td>
      <td>0.01394</td>
      <td>0.06134</td>
      <td>0.626</td>
      <td>...</td>
      <td>0.04368</td>
      <td>0.09403</td>
      <td>0.01929</td>
      <td>19.085</td>
      <td>0.458359</td>
      <td>0.819521</td>
      <td>-4.075192</td>
      <td>0.335590</td>
      <td>2.486855</td>
      <td>0.368674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>116.682</td>
      <td>131.111</td>
      <td>111.555</td>
      <td>0.01050</td>
      <td>0.00009</td>
      <td>0.00544</td>
      <td>0.00781</td>
      <td>0.01633</td>
      <td>0.05233</td>
      <td>0.482</td>
      <td>...</td>
      <td>0.03590</td>
      <td>0.08270</td>
      <td>0.01309</td>
      <td>20.651</td>
      <td>0.429895</td>
      <td>0.825288</td>
      <td>-4.443179</td>
      <td>0.311173</td>
      <td>2.342259</td>
      <td>0.332634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>116.676</td>
      <td>137.871</td>
      <td>111.366</td>
      <td>0.00997</td>
      <td>0.00009</td>
      <td>0.00502</td>
      <td>0.00698</td>
      <td>0.01505</td>
      <td>0.05492</td>
      <td>0.517</td>
      <td>...</td>
      <td>0.03772</td>
      <td>0.08771</td>
      <td>0.01353</td>
      <td>20.644</td>
      <td>0.434969</td>
      <td>0.819235</td>
      <td>-4.117501</td>
      <td>0.334147</td>
      <td>2.405554</td>
      <td>0.368975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>116.014</td>
      <td>141.781</td>
      <td>110.655</td>
      <td>0.01284</td>
      <td>0.00011</td>
      <td>0.00655</td>
      <td>0.00908</td>
      <td>0.01966</td>
      <td>0.06425</td>
      <td>0.584</td>
      <td>...</td>
      <td>0.04465</td>
      <td>0.10470</td>
      <td>0.01767</td>
      <td>19.649</td>
      <td>0.417356</td>
      <td>0.823484</td>
      <td>-3.747787</td>
      <td>0.234513</td>
      <td>2.332180</td>
      <td>0.410335</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>190</th>
      <td>174.188</td>
      <td>230.978</td>
      <td>94.261</td>
      <td>0.00459</td>
      <td>0.00003</td>
      <td>0.00263</td>
      <td>0.00259</td>
      <td>0.00790</td>
      <td>0.04087</td>
      <td>0.405</td>
      <td>...</td>
      <td>0.02745</td>
      <td>0.07008</td>
      <td>0.02764</td>
      <td>19.517</td>
      <td>0.448439</td>
      <td>0.657899</td>
      <td>-6.538586</td>
      <td>0.121952</td>
      <td>2.657476</td>
      <td>0.133050</td>
    </tr>
    <tr>
      <th>191</th>
      <td>209.516</td>
      <td>253.017</td>
      <td>89.488</td>
      <td>0.00564</td>
      <td>0.00003</td>
      <td>0.00331</td>
      <td>0.00292</td>
      <td>0.00994</td>
      <td>0.02751</td>
      <td>0.263</td>
      <td>...</td>
      <td>0.01879</td>
      <td>0.04812</td>
      <td>0.01810</td>
      <td>19.147</td>
      <td>0.431674</td>
      <td>0.683244</td>
      <td>-6.195325</td>
      <td>0.129303</td>
      <td>2.784312</td>
      <td>0.168895</td>
    </tr>
    <tr>
      <th>192</th>
      <td>174.688</td>
      <td>240.005</td>
      <td>74.287</td>
      <td>0.01360</td>
      <td>0.00008</td>
      <td>0.00624</td>
      <td>0.00564</td>
      <td>0.01873</td>
      <td>0.02308</td>
      <td>0.256</td>
      <td>...</td>
      <td>0.01667</td>
      <td>0.03804</td>
      <td>0.10715</td>
      <td>17.883</td>
      <td>0.407567</td>
      <td>0.655683</td>
      <td>-6.787197</td>
      <td>0.158453</td>
      <td>2.679772</td>
      <td>0.131728</td>
    </tr>
    <tr>
      <th>193</th>
      <td>198.764</td>
      <td>396.961</td>
      <td>74.904</td>
      <td>0.00740</td>
      <td>0.00004</td>
      <td>0.00370</td>
      <td>0.00390</td>
      <td>0.01109</td>
      <td>0.02296</td>
      <td>0.241</td>
      <td>...</td>
      <td>0.01588</td>
      <td>0.03794</td>
      <td>0.07223</td>
      <td>19.020</td>
      <td>0.451221</td>
      <td>0.643956</td>
      <td>-6.744577</td>
      <td>0.207454</td>
      <td>2.138608</td>
      <td>0.123306</td>
    </tr>
    <tr>
      <th>194</th>
      <td>214.289</td>
      <td>260.277</td>
      <td>77.973</td>
      <td>0.00567</td>
      <td>0.00003</td>
      <td>0.00295</td>
      <td>0.00317</td>
      <td>0.00885</td>
      <td>0.01884</td>
      <td>0.190</td>
      <td>...</td>
      <td>0.01373</td>
      <td>0.03078</td>
      <td>0.04398</td>
      <td>21.209</td>
      <td>0.462803</td>
      <td>0.664357</td>
      <td>-5.724056</td>
      <td>0.190667</td>
      <td>2.555477</td>
      <td>0.148569</td>
    </tr>
  </tbody>
</table>
<p>195 rows × 22 columns</p>
</div>




```python
# randomly fill n/a values to randomly as it is discrete
y = fullData.status.fillna(randint(0,1))
y
```




    0      1
    1      1
    2      1
    3      1
    4      1
          ..
    190    0
    191    0
    192    0
    193    0
    194    0
    Name: status, Length: 195, dtype: int64



### Applied Models
- chosen model making algorithms


```python
# TODO Apply models
```

# Results
TODO

### Accuracy


```python
# TODO show results and visualizations
```

# CONTRIBUTORS
Members of `CSE422`, section `1`, project team `Elytra`:
- Mohammad Rafid Hamid, Student ID: `20101491`
- Nihad
- Shama

# REFERENCES


<a id="cite1">[1]</a> R. Prashanth, Sumantra Dutta Roy,
Early detection of Parkinson’s disease through patient questionnaire and predictive modelling,
International Journal of Medical Informatics,
Volume 119,
2018,
Pages 75-87,
ISSN 1386-5056,
https://doi.org/10.1016/j.ijmedinf.2018.09.008.

<a id="cite2">[2]</a> 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection',
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.
BioMedical Engineering OnLine 2007, 6:23,
June 2007
https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection
