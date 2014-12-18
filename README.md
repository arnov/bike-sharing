Kaggle bike sharing prediction
============

This code can be used to make predictions for the Kaggle bike-sharing competition. See https://www.kaggle.com/c/bike-sharing-demand

The code is written in Python and makes use of a couple of libraries which are really helpful for machine learning and data analysis. For instance Pandas is used for dataframe support, Scikit-learn for its machine learning algorithms and Matplotlib for visualizations.

Ipython Notebooks are a nice way of combining presentation and code and interactively running parts of the program. A static version of the notebook in this repo can be found here http://nbviewer.ipython.org/github/arnov/bike-sharing/blob/master/Bike-Sharing-Analysis.ipynb

Installation
============

Requirements:

- Python2.7
- Virtualenv (recommended)


```
> virtualenv env
> source env/bin/activate
> pip install -r requirements.txt  
# Might take a while, especially installing scipy and numpy can take long)
```

Running
============

Use ```ipython notebook``` to start the interactive notebook or use ```python prediction.py``` to run the prediction script.
