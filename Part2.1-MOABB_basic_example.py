
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')

import moabb
from moabb.datasets import BNCI2014001
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


# In[2]:

dataset = BNCI2014001()

paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset], overwrite=True)

pipeline = make_pipeline(CSP(n_components=8), LDA())

results = evaluation.process({'csp+lda':pipeline}) 


# In[5]:

results


# In[6]:

fig, ax = plt.subplots(figsize=(8,7))
results["subj"] = results["subject"].apply(str)
sns.barplot(x="score", y="subj", hue='session', data=results, orient='h', palette='viridis', ax=ax)
#sns.catplot(kind='bar', x="score", y="subj", hue='session', data=results, orient='h', palette='viridis')
fig.show()


# In[ ]:



