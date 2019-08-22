
# coding: utf-8

# In[6]:

get_ipython().magic('matplotlib inline')

import moabb
from moabb.datasets import BNCI2014001, Weibo2014, Zhou2016
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

import matplotlib.pyplot as plt
import seaborn as sns

import mne
mne.set_log_level("CRITICAL")

moabb.set_log_level('info')
import warnings
warnings.filterwarnings("ignore")


# In[7]:

datasets = [BNCI2014001(), Weibo2014(), Zhou2016()]

paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)

pipelines = {}
pipelines['csp+lda'] = make_pipeline(CSP(n_components=8), LDA())

parameters = {'C': np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel='linear'), parameters)
pipelines['tgsp+svm'] = make_pipeline(Covariances('oas'), TangentSpace(metric='riemann'), clf)

clf = MDM(metric='riemann')
pipelines['MDM'] = make_pipeline(Covariances('oas'), clf)

results = evaluation.process(pipelines) 


# In[8]:

results


# In[21]:

results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
g = sns.catplot(kind='bar', x="score", y="subj", hue="pipeline", col="dataset", height=12, aspect=0.5, data=results, orient='h', palette='viridis')


# In[36]:

g = sns.catplot(kind="box", x="score", y="pipeline", height=, aspect=1.5, data=results, orient='h', palette='viridis')


# In[37]:

g.fig.


# In[ ]:



