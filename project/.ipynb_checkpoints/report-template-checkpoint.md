# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Jai Ranjan Singh Gusain

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
It was mentioned in the notebook that Kaggle would reject the submissions if my predictions were negative.
Applied a lambda function to make sure each value in the predictions was positive.

### What was the top ranked model that performed?
The best model was the WeightedEnsembleL2 which had a root mean squared error of 33.383077 and a kaggle submission score 0.57057.
This was achieved with the hyperparameter tuning of higher level parameters such as max_bag_folds, num_stack_levels, time_limit and individual model parameters of Neural Network and Lightgbm

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The histogram of the target variable showed that the target variable is not normally distributed hence linear regression would not make a good choice
for the modelling technique. Therefore using a methodology like AutoGluon makes sense since it tries out different non linear modelling techniques.
![Histogram.png](Histogram.png)

AutoGluon was already extracting year, month, day of week from the datetime variable.
Hour of the day for demand or booking extracted from the datetime could be an important feature since we are predicting hourly count of rental bookings. 
Example: train['demand_hour'] = train['datetime'].dt.hour

The correlation was plotted using the heatmap of seaborn package which showed the strength of the relation between variables.
Hour was the variables with the highest correlation with the target variable count.
Other high correlation variables were temperature , atemp, humidity and datetime

### How much better did your model preform after adding additional features and why do you think that is?
The performance imporoved drastically after adding the hour variable.
The rootmean squared error went down to 34.34 from 53.04 and the kaggle score also improved from 1.80205 to 0.58770.
Intuitively the variable made sense since we are predicting the hourly bike demand.
The correlation of the hour vairable was also the highest with the target at 0.40 and was substantially lower with other predictor variables. 
![Correlation_Matrix.png](Correlation_Matrix.png)

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The model did slightly better after hyperparameter tuining.
The rootmean squared error went down to 33.38 from 34.34, and the kaggle score also improved from 0.58770 to 0.57057.

### If you were given more time with this dataset, where do you think you would spend more time?
If given more time would like to follow AutoGluon recommendation of just using presets="best_quality" and not imposing any time limits on the training of the models in hyperparameters.
Hyperparameter tuning could also be explored in increasing the number of trials so we can try more combination of models and also tune the individual parameters of catboost and xgboost while also expand the tuning features of both nueral networks and lightgbm and expand the search space. 
More improvement on the score can be made by tuning hyperparamets more.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|hpo4|hpo5|score|
|--|--|--|--|--|--|--|
|initial|max_bag_folds=8|num_stack_levels=3|time_limit=600|presets=best_quality|Individual Model Hyperparameters=[None]|1.80091|
|add_features|max_bag_folds=10|num_stack_levels=4|time_limit=600|presets=best_quality|Individual Model Hyperparameters=[None]|0.74380|
|hpo|max_bag_folds=9|num_stack_levels=3|time_limit=800|presets=None(default)|Individual Model Hyperparameters=[N.Network,LGBM]|0.57057|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](model_test_score.png)

## Summary
This project provided an interesting opportunity to tackle a challenging problem where the target variable (count) is not normally distributed. For this dataset, non-linear approaches proved to be more effective than simple linear models. AutoGluon served as a robust framework to explore and optimize various modeling techniques.

Through iterative experimentation:

A Kaggle score of 1.8 was achieved initially.
Key improvements were made by:
Adding the hour feature, which had the highest correlation (0.40) with the target.
Treating weather and season as categorical variables instead of continuous.
Tuning high-level parameters like max_bag_folds and num_stack_level.
Increasing the time_limit to allow more thorough model exploration.
Fine-tuning individual model parameters to optimize base models for the WeightedEnsembleL2.
These improvements led to a final Kaggle score of 0.57. Future work could include expanding the hyperparameter search space, experimenting with additional feature interactions, and using transformations to handle the fat-tailed distribution of the target variable.





