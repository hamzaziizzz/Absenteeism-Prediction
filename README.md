# Absenteeism Prediction

## Project Description
This project will address and predict the *absenteeism* at a company during work hours.

***Absenteeism***: *Absence from work during normal working hours, resulting in temporary incapacity to execute regular working activity.*

### Problems
-	higher competitiveness → increased pressure
-	unachievable business goals → raised stress levels
-	elevated risk of becoming unemployed → raised stress levels

can raise the problems of being excessively absent at work hours.

### Questions
-	Based on what information should we predict whether an employee is expected to be absent or not?
-	How would we measure absenteeism?

### Purpose
Explore whether a person presenting certain characteristics is expected to be away from work at some point in time or not.

We want know for how many working hours an employee could be away from work based on:
- How far they live from their workplace?
- How many children and pets they have?
- Do they have higher education?

And so on…

## Project Requirements
Install the following Python libraries on your local system:
```
pip install numpy
```

```
pip install pandas
```

```
pip install scikit-learn
```

or you can install using *requirements.txt* file
```
pip install -r requirements.txt
```

## How to Run the Jupyter Source Files
1) Clone the repository on your local system
2) Run ```jupyter notebook``` command in the directory where you have cloned the repository. Make sure [Anaconda](https://www.anaconda.com/products/distribution) or [Jupyter Labs](https://jupyter.org/install) is already installed on your local system.
3) Run [Absenteeism Raw CSV Data Preprocessing.ipynb](https://github.com/hamzaziizzz/Absenteeism-Prediction/blob/main/Absenteeism%20Raw%20CSV%20Data%20Preprocessing.ipynb) in the jupyter lab. A [Absenteeism_preprocessed.csv](https://github.com/hamzaziizzz/Absenteeism-Prediction/blob/main/Absenteeism_preprocessed.csv) will be created which contains the information for building the Logistic Regression Model.
4) Run [Absenteeism - Logistic Regression.ipynb](https://github.com/hamzaziizzz/Absenteeism-Prediction/blob/main/Absenteeism%20-%20Logistic%20Regression.ipynb) to build the model. This will save the model which can be implemented to predict the *Excessive Absenteeism*.

### Integrating the Model
1) Create a new folder say Integration.
2) Make sure this folder contains 5 files for sure - *Absenteeism_new_data.csv*, *absenteeism_module.py*, *absenteeism_model*, *absenteeism_scaler*, *Absenteeism-Integration.ipynb*
3) Run [Absenteeism-Integration.ipynb](https://github.com/hamzaziizzz/Absenteeism-Prediction/blob/main/Integration/Absenteeism-Integration.ipynb) and the prediction values will be saved in a file named as *Absenteeism_predictions.csv*

*Alternatively, you can directly go into the Integration folder and run the jupyter file to use the pre-trained model.*
