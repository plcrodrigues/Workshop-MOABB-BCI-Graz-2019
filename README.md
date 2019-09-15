# 10:15 - 11:00 : Classification methods in BCI   
- Marco's presentation

# 11:00 - 12:00 : MOABB Project
- Motivation for creating/using MOABB
- Present MNE
- Present Scikit-learn (not all published estimators are available in there)
- PyRiemann is an example of DYI classifier with scikit-learn template

# 12:00 - 13:00 : Installation and basic setup

- An e-mail to all people participating in the workshop with the instructions to install MOABB, scikit-learn, MNE, pyriemann, etc., etc. (which datasets to download as well)
- For those that are not comfortable in installing by themselves, we will propose a virtual machine with EVERYTHING installed and downloaded. The user will have less margin of control, but we will be sure that it works.

# 14:30 - 15:15 : Hands-on: easy benchmark

Part I : 
Basic concepts on Machine Learning classification, suppose we already have the trials (no mention about MOABB) 
Dataset downloaded by hand and loaded via scipy.io.loadmat 
Dataset BNCI2014001 ? 

Part II :
Create a scikit-learn notebook with step-by-step for classification 
- Show how to use MOABB for downloading, filtering, epoching data
- Analyze the signals without classification (MOABB can do this)
- Use scikit-learn/pyriemann in a step-by-step way for classification
    + Explain cross-validation procedure ? KFold ?
    + Do the steps one by one or use a make_pipeline 
- Use moabb's evalutation procedure with a pipeline created above 

# 15:15-16:30 : Hands-on: write your own classifier ! Write your own dataset !

- What are the methods and paradigm for creating a classifier/estimator on scikit-learn (fit, predict, etc)
- How to create a dataset in MOABB
- Small local "competition" for testing different pipelines and getting scores on certain datasets



