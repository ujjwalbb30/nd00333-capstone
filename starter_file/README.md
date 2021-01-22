*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

I will be using the 'Heart Failure Clinical Data' which consists of 12 features ( age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time ) which can be used to predict mortality by heart failure. There are total of 299 input rows in the dataset with 0 null entries. I got this dataset from KAGGLE and it can be accessed through the following link:

SOURCE : https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
A machine learning classification model on this dataset will be helpful for early detection of people with cardiovascular disease or those who are at high risk of cardiovascular disease.

The 12 features are as follows:

(1) age

(2) anaemia i.e. decrease of red blood cells or hemoglobin (boolean)

(3) creatining_phosphokinase i.e. level of the CPK enzyme in the blood (mcg/L)

(4) diabetes i.e. if the patient has diabetes or not (boolean)

(5) ejection_fraction i.e. percentage of blood leaving the heart at each contraction (percentage)

(6) high_blood_pressure i.e. if the patient has hypertension (boolean)

(7) platelets i.e. platelets in the blood (kiloplatelets/mL)

(8) serum_creatinine i.e. level of serum creatinine in the blood (mg/dL)

(9) serum_sodium i.e. level of serum sodium in the blood (mEq/L)

(10) sex i.e. woman or man (binary)

(11) smoking i.e. if the patient smokes or not (boolean)

(12) time i.e. follow-up period (days)

We will be predicting the following output:

DEATH_EVENT i.e if the patient deceased during the follow-up period (boolean)

### Access
*TODO*: Explain how you are accessing the data in your workspace.

HYPERDRIVE RUN : We are first registering the dataset with 

AUTOML RUN : We are accessing the dataset using TabularDatasetFactory by providing the url to raw form of data. The url is : "https://raw.githubusercontent.com/ujjwalbb30/nd00333-capstone/ujjwalbb30-patch-1/heart_failure_clinical_records_dataset.csv"

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

for automl settings, I will be using the following parameters:

(1) experiment_timeout_minutes : It is the amount of time that the experiment will run upto. I will input it as 30 minutes which means the the experiment will exit after 30 minutes ( if it doesn't find the best run within 30 minutes and exit on its own ) and will give out the best result found during that time.

(2) max_concurrent_iterations : It is the maximum number of iterations to be executed in parallel. I will input it as '5' iterations. 5 concurrent iterations will help in quickly executing the tasks of experiment and it will also not load the compute target too much for computation.

(3) primary_metric : This is the metric that will be optimized by Automated Machine Learning for model selection. I will use 'AUC_weighted' as 'primary_metric' parameter. AUC means the area under the Receiver Operating Characteristic Curve which plots the relationship between true positive rate and false positive rate. Since our dataset doesn't have high class imbalance, we can use ROC method for judging the performance of a model. I will use AUC_weighted in order to mitigate the effects of whatever little imbalance is there in the dataset. AUC_weighted is the arithmetic mean of the score for each class, weighted by the number of true instances in each class.

for automl configuration, I will be using the following parameters:

(1) compute_target : It is the compute target on which we will run our Azure Machine Learning experiment. Since I have created a compute target named as 'compute_target' for this purpose, I will input it as the 'compute_target' parameter.

(2) task : I want to make a classification model that can predict whether the patient is at a high risk of cardiovascular disease or not. Hence, I will input 'classification' as 'task' parameter.

(3) training_data : It is the training dataset to be used for the experiment. I will use 'dataset' (the registered dataset imported above for running this experiment) as 'training_data' parameter. importing training dataset means the output columns will be included and its name will be entered in 'label_column_name'.

(4) label_column_name : It is the name of the output column present in the training dataset. I will enter 'DEATH_EVENT' as 'label_column_name' parameter.

(5) path : This is the full path to the Azure Machine learning project folder. Hence, I will input './pipeline-project' as 'path' parameter.

(6) enable_early_stopping : we can choose to terminate the experiment if the score stops improving in the short term. I will enter 'True' as 'enable_early_stopping' parameter.

(7) featurization : It is the option to featurize the dataset i.e. whether we want the Azure to do it automatically or we want to turn it off or we want some customized featurization step. I will input 'auto' in the 'featurization' parameter as I want Azure to featurize the dataset automatically.

(8) debug_log : it is the log file in which debug information is written. I am entering 'automl_errors.log' as 'debug_log' parameter.

(9) n_cross_validations : It is the number of cross validations performed. I will input it as '5' since the input rows is way lower than 1000 and 5 cross validations will not be very computation expensive.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

The model I chose for the classification purpose of the heart failure clinical dataset is explained below. I have tried to explain the pipeline architecture, hyperparameter tuning, and classification algorithm:

(1) The pipeline is created using HyperDriveConfig which requires an estimator, an early termination policy, a parameter sampler, a primary metric name, a primary metric goal and a value for maximum total runs.

(2) A parameter sampler is created using RandomParameterSampling which generates a set values (equal to value of maximum total runs) of C and max_iter to be used in child runs of experiment. For C, a continuous set of values ranging from 0.0005 to 1.0 is used and for max_iter, a discrete set of values is used which includes 50,100,150,200 and 250.

(3) BanditPolicy is used as early termination policy with evalution_interval as 5, slack_amount as 0.2 and delay_evalution as 5. This means if Run X is the currently best performing run with an accuracy of 0.9 after 5 intervals, then any run with an accuracy less than 0.7 (0.9 - 0.2) after 5 iterations will be terminated, and the delay_evaluation will delay the first termination policy evaluation for 5 sequences.

(4) An SKLearn estimator is used with train.py as training script (in which the features and labels are first segregated and the dataset is split into train and test using train_test_split module) for Scikit-Learn experiments which trains a Logistic Regression model on heart failure clinical data with varying sets of values of C and max_iter supplied by the parameter sampler (RandomParameterSampling in our case).

The benefits of the parameter sampler I chose are mentioned below:

The parameter sampler chosen is RandomParameterSampling which selects hyperparameter values randomly from the defined search space. RandomParameterSampling results in good results without consuming too much time.

The benefits of the early stopping policy I chose are given below:

The early stopping policy chosen is BanditPolicy with evalution_interval as 5, slack_amount as 0.2 and delay_evalution as 5. This means if Run X is the currently best performing run with an accuracy of 0.9 after 5 intervals, then any run with an accuracy less than 0.7 (0.9 - 0.2) after 5 iterations will be terminated, and the delay_evaluation will delay the first termination policy evaluation for 5 sequences. This means I will not lose promising jobs and also the jobs with poor performance will be terminated early hence saving computation time and costs.

The parameters of HyperDriveConfig are explained as below:

(1) estimator: It is the model estimator to be used to run the model. I have defined an SKLearn estimator below as 'est' and I will use it as 'estimator' parameter.

(2) hyperparameter_sampling: It is the sampler that will create the instance of hyperparameters to be used for each sample run. I have defined a RandomParameterSampling below as 'param_sampling' and I will use it as 'hyperparameter_sampling' parameter.

(3) policy: It is the early termination policy that will be used to terminate the experiment if no improvement in primary metric is witnessed after some runs. I have defined a BanditPolicy below as 'et_policy' and I will use it as 'policy' parameter.

(4) primary_metric_name: it is the name of the metric on the basis of which performance of different models will be judged. I will be using 'AUC_weighted' as the 'primary_metric_name' parameter. AUC means the area under the Receiver Operating Characteristic Curve which plots the relationship between true positive rate and false positive rate. Since our dataset doesn't have high class imbalance, we can use ROC method for judging the performance of a model. I will use AUC_weighted in order to mitigate the effects of whatever little imbalance is there in the dataset. AUC_weighted is the arithmetic mean of the score for each class, weighted by the number of true instances in each class.

(5) primary_metric_goal: In order to get the best model for our classification task, my goal is to maximize the 'AUC_weighted' metric hence I will enter 'PrimaryMetricGoal.MAXIMIZE'as 'primary_metric_goal' parameter.

(6) max_total_runs: It is the maximum number of child runs that will be executed in the experiment to find the best model for the task intended. I will enter '25' as the 'max_total_runs' parameter which will produce a good and acceptable result in less amount of time.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
