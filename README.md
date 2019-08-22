# TODO

## 09:15 - 10:00 : Classification methods in BCI   
- Marco's presentation

# 10:15 - 11:00 : MOABB Project
- Motivation for creating/using MOABB
- Present MNE
- Present Scikit-learn (not all published estimators are available in there)
- PyRiemann is an example of DYI classifier with scikit-learn template

## 11:00 - 12:00 : Installation and basic setup
- An e-mail to all people participating in the workshop with the instructions to install MOABB, scikit-learn, MNE, pyriemann, etc., etc. (which datasets to download as well)
- For those that are not comfortable in installing by themselves, we will propose a virtual machine with EVERYTHING installed and downloaded. The user will have less margin of control, but we will be sure that it works.

## 13:30 - 14:15 : Hands-on: easy benchmark

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

## 14:30-16:00 : Hands-on: write your own classifier ! Write your own dataset !

- What are the methods and paradigm for creating a classifier/estimator on scikit-learn (fit, predict, etc)
- Small local "competition" for testing different pipelines and getting scores on certain datasets

--------------

1) Installation procedures :
    - Write an e-mail for the installation at home
    - Create an environment conda for the installation at home
    - Otherwise, the list of packages in .txt
    - Ask people to install VirtualBox on their PC
    - Create the virtual machines in Ubuntu 
    - Choose and download the datasets

2) Presentation about MOABB in the morning

3) Codes for 13:30 - 14:15 : Add commentaries to the codes
    - Part1.1-MOABB_basic_example.ipynb
    - Part2.1-MOABB_basic_example.ipynb
    - Part2.2-MOABB_basic_example.ipynb
    - Part2.3-MOABB_basic_example.ipynb

4) Codes for 14:30 - 16:00 :
    - Write your own classifier
    - Write your own dataset



