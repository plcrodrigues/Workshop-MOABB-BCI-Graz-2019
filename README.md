# TODO

## 09:15 - 10:00 : Classification methods in BCI   
- Marco's presentation

# 10:15 - 11:00 : MOABB Project
- Motivation for creating/using MOABB
- Present MNE
- Present Scikit-learn (not all published estimators are available in there)
- PyRiemann is an example of DYI classifier with scikit-learn template

## 11:00 - 12:00 : Installation and basic setup
[TODO Deux] Check if can create a folder with all the packages physically (not just the .yml file for installation)

Take a look at this solution proposed by Raphael Bacher
https://conda.github.io/conda-pack/

## 13:30 - 14:15 : Hands-on: easy benchmark

Part I : 
Basic concepts on Machine Learning classification, suppose we already have the trials (no mention about MOABB) 
AlphaWaves : https://github.com/plcrodrigues/py.ALPHA.EEG.2017-GIPSA 

Part II :
Create a scikit-learn notebook with step-by-step for classification 
- Show how to use MOABB for downloading, filtering, epoching data
- Analyze the signals without classification (MOABB can do this)
- Use scikit-learn/pyriemann in a step-by-step way for classification
    + Explain cross-validation procedure ? KFold ?
    + Do the steps one by one or use a make_pipeline 
- Use moabb's evalutation procedure with a pipeline created above 

[TODO Pedro] Choose the datasets we want to use and have a copy on USB (do not depend on the internet) 
[TODO Sylvain] Find the scikit-learn notebook explaining each step for Machine Learning

## 14:30-16:00 : Hands-on: write your own classifier !
- What are the methods and paradigm for creating a classifier/estimator on scikit-learn (fit, predict, etc)
- Small local "competition" for testing different pipelines and getting scores on certain datasets

