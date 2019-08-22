
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

import moabb
from moabb.datasets import BNCI2014001, Weibo2014, Zhou2016
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP

import matplotlib.pyplot as plt
import seaborn as sns

moabb.set_log_level('info')
import warnings
warnings.filterwarnings("ignore")

import mne
mne.set_log_level("CRITICAL")


# In[ ]:

datasets = [Zhou2016(), BNCI2014001(), Weibo2014()]

paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)

pipeline = make_pipeline(CSP(n_components=8), LDA())

results = evaluation.process({'csp+lda':pipeline}) 


# In[5]:

results


# In[11]:

results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
g = sns.catplot(kind='bar', x="score", y="subj", col="dataset", data=results, orient='h', palette='viridis')


# In[ ]:



