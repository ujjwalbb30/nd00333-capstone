# CAPSTONE PROJECT - AZURE MACHINE LEARNING ENGINEER

In this capstone project, I performed a machine learning classification task on a dataset ( 'Heart Failure Clinical Data' from KAGGLE, discussed in 'Dataset' section below ). I created two models for this purpose and then compared their performace on the basis of the scoring metric ( 'AUC_weighted' in my case ).

The first model was created using Automated ML and the best model produced was VotingEnsemble model with a metric score of 0.9196.

The second model was created using Hyperdrive and two hyperparameters were chosen for tuning and performing various iterations. I chose C (Inverse of Regularization strength) and max_iter (Maximum number of iterations taken for the solvers to converge) as my hyperparameters. The best performing Hyperdrive model was a Logistic Regression model with parameter value of C and max_iter as '0.5633339376963704' and '100' respectively. It performed with an accuracy of 0.7575757575757576.

Since Automated ML produced model with a higher metric score, I deployed that model as an Azure Container Instance (ACI). I tested this deployed model by sending a random data sample as a request and the model responded with an output ( '0' in my case meaning the patient did not die during the follow-up period ) which demonstrated its successful deployment.

## Project Set Up and Installation

I took almost 8 Azure lab sessions to create this project. With the help of these lab sessions, I created a github repository of training script (train.py), scoring script (entry_script.py), AutoML python notebook (automl.ipynb) and Hyperdrive python notebook (hyperparameter_tuning.ipynb) for smooth execution of this project in future. I also downloaded the dataset from Kaggle and uploaded it in this repository so that I can use it as per my convenience and have all the necessary files in one place. In my final lab session, I downloaded this repository and then uploaded these necessary files in Azure ML Studio and performed all the tasks with ease.

## Dataset

### Overview

I used the 'Heart Failure Clinical Data' which consists of 12 features ( age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time ) which can be used to predict mortality by heart failure. There are a total of 299 input rows in the dataset with 0 null entries. I got this dataset from KAGGLE and it can be accessed through the following link:

SOURCE : https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

### Task

A machine learning classification model on this dataset is created with the help of Azure and it will be helpful for early detection of people with cardiovascular disease or those who are at high risk of cardiovascular disease.

The 12 features are as follows:

(1) age

(2) anaemia i.e. decrease of red blood cells or hemoglobin (boolean)

(3) creatinine_phosphokinase i.e. level of the CPK enzyme in the blood (mcg/L)

(4) diabetes i.e. if the patient has diabetes or not (boolean)

(5) ejection_fraction i.e. percentage of blood leaving the heart at each contraction (percentage)

(6) high_blood_pressure i.e. if the patient has hypertension (boolean)

(7) platelets i.e. platelets in the blood (kiloplatelets/mL)

(8) serum_creatinine i.e. level of serum creatinine in the blood (mg/dL)

(9) serum_sodium i.e. level of serum sodium in the blood (mEq/L)

(10) sex i.e. woman or man (binary)

(11) smoking i.e. if the patient smokes or not (boolean)

(12) time i.e. follow-up period (days)

We predicted the following output:

DEATH_EVENT i.e if the patient deceased during the follow-up period (boolean)

### Access

HYPERDRIVE RUN : I first registered the dataset with key = 'heart-failure-clinical-data' and description_text = 'heart failure predictions'. Then, I accessed it using Workspace library with following command:

dataset = ws.datasets[key]        ### where ws is the initialized workspace ( please see hyperparameter_tuning.ipynb for more details )

AUTOML RUN : I accessed the dataset using TabularDatasetFactory library by providing the url to raw form of data ( through my github repository ). The url to data ( in raw form ) is as follows:

"https://raw.githubusercontent.com/ujjwalbb30/nd00333-capstone/ujjwalbb30-patch-1/heart_failure_clinical_records_dataset.csv"

the command I used to access the dataset is as follows:

dataset = TabularDatasetFactory.from_delimited_files("https://raw.githubusercontent.com/ujjwalbb30/nd00333-capstone/ujjwalbb30-patch-1/heart_failure_clinical_records_dataset.csv")

## Automated ML

for automl settings, I used the following parameters:

(1) experiment_timeout_minutes : It is the amount of time that the experiment will run upto. I will input it as 30 minutes which means the the experiment will exit after 30 minutes ( if it doesn't find the best run within 30 minutes and exit on its own ) and will give out the best result found during that time.

(2) max_concurrent_iterations : It is the maximum number of iterations to be executed in parallel. I will input it as '5' iterations. 5 concurrent iterations will help in quickly executing the tasks of experiment and it will also not load the compute target too much for computation.

(3) primary_metric : This is the metric that will be optimized by Automated Machine Learning for model selection. I will use 'AUC_weighted' as 'primary_metric' parameter. AUC means the area under the Receiver Operating Characteristic Curve which plots the relationship between true positive rate and false positive rate. Since our dataset doesn't have high class imbalance, we can use ROC method for judging the performance of a model. I will use AUC_weighted in order to mitigate the effects of whatever little imbalance is there in the dataset. AUC_weighted is the arithmetic mean of the score for each class, weighted by the number of true instances in each class.

for automl configuration, I used the following parameters:

(1) compute_target : It is the compute target on which we will run our Azure Machine Learning experiment. Since I have created a compute target named as 'compute_target' for this purpose, I will input it as the 'compute_target' parameter.

(2) task : I want to make a classification model that can predict whether the patient is at a high risk of cardiovascular disease or not. Hence, I will input 'classification' as 'task' parameter.

(3) training_data : It is the training dataset to be used for the experiment. I will use 'dataset' (the registered dataset imported above for running this experiment) as 'training_data' parameter. importing training dataset means the output columns will be included and its name will be entered in 'label_column_name'.

(4) label_column_name : It is the name of the output column present in the training dataset. I will enter 'DEATH_EVENT' as 'label_column_name' parameter.

(5) path : This is the full path to the Azure Machine learning project folder. Hence, I will input './pipeline-project' as 'path' parameter.

(6) enable_early_stopping : We can choose to terminate the experiment if the score stops improving in the short term. I will enter 'True' as 'enable_early_stopping' parameter.

(7) featurization : It is the option to featurize the dataset i.e. whether we want the Azure to do it automatically or we want to turn it off or we want some customized featurization step. I will input 'auto' in the 'featurization' parameter as I want Azure to featurize the dataset automatically.

(8) debug_log : It is the log file in which debug information is written. I am entering 'automl_errors.log' as 'debug_log' parameter.

(9) enable_onnx_compatible_models : Setting it to 'True' will help in retrieving the best automl model in ONNX format.

### Results

At first, AutoML checked for the 'Class Balancing Detection' and 'High Cardinality Feature Detection' for the dataset. Both of these checks were passed by the dataset meaning that there was no class imbalance and no high cardinality feature present. AutoML performed 52 iterations out of which it produced VotingEnsemble as the best model with a metric score of 0.9196.

VotingEnsemble is an ensemble model which combines multiple models to improve machine learning results. It does so by predicting output on the weighted average of predicted class probabilities. So, the hyperparameters for VotingEnsemble pipeline are the ensemble_iterations and their respective weights. According to my code outputs, out of the 52 iterations ran by AutoML, iteration 33 ('RandomForest') , iteration 23 ('RandomForest'), iteration 46 ('GradientBoosting'), iteration 48 ('GradientBoosting'), iteration 12 ('RandomForest'), iteration 20 ('RandomForest') and iteration 7 ('ExtremeRandomTrees') were chosen to be the ensemble iterations and 0.16666666666666666, 0.25, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666, 0.16666666666666666 and 0.08333333333333333 were their respective weights.

In order to complete all other tasks along with automl run, I purposely kept the 'experiment_timeout_minutes' parameter equal to 30 minutes. It implies that my automl run was only allowed 30 minutes to run before the experiment terminated and the model with the best metric score was produced. By increasing the 'experiment_timeout_minutes' there is a possibility that a model with a better metric score can be produced using automl. 

Screenshots of RunDetails widget along with best model and its parameters:

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/1.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/2.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/3.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/4.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/5.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/6.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/7.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/8.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/9.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/10.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/11.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/12.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/13.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/14.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/15.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/25.PNG)

## Hyperparameter Tuning

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

For Hyperdrive Run, two hyperparameters were chosen for tuning and performing various iterations. I chose C (Inverse of Regularization strength) and max_iter (Maximum number of iterations taken for the solvers to converge) as my hyperparameters. The best performing Hyperdrive model was a Logistic Regression model with parameter value of C and max_iter as '0.5633339376963704' and '100' respectively. It performed with an accuracy of 0.7575757575757576.

In order to complete all other tasks along with hyperdrive run, I purposely kept the range of 'C', 'max_iter' and 'max_total_runs' limited. By widening the range of 'C' and 'max_iter', scope of combinations of values of 'C' and 'max_iter' increases which might result in a hyperdrive model with a better metric score. Increasing the value of 'max_total_runs' will complement the widening of range of 'C' and 'max_iter' as it will allow more such combination of values to be tested.

Screenshots of RunDetails widget along with best model and its parameters:

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/16.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/17.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/18.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/19.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/20.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/21.PNG)

## Model Deployment

Since Automated ML produced model with a higher metric score i.e. 0.9196, I deployed that model as an Azure Container Instance (ACI). I tested this deployed model by sending a random data sample as request and the model responded with an output ( '0' in my case meaning the patient did not die during the follow-up period ) which demonstrated its successful deployment.

The deployed model is a VotingEnsemble model produced by Automated ML ( which ran 52 iterations in total ). VotingEnsemble is an ensemble model which combines multiple models to improve machine learning results. It does so by predicting output on the weighted average of predicted class probabilities. So, the hyperparameters for VotingEnsemble pipeline are the ensemble_iterations and their respective weights. According to my code outputs, out of the 52 iterations ran by AutoML, iteration 33 ('RandomForest') , iteration 23 ('RandomForest'), iteration 46 ('GradientBoosting'), iteration 48 ('GradientBoosting'), iteration 12 ('RandomForest'), iteration 20 ('RandomForest') and iteration 7 ('ExtremeRandomTrees') were chosen to be the ensemble iterations and 0.16666666666666666, 0.25, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666, 0.16666666666666666 and 0.08333333333333333 were their respective weights.

In order to query the endpoint with a sample input, I created a random test sample and sent it in JSON form as a request to the scoring script ( entry_script.py) of the deployed model. The test sample was created in the following way :

data = {"data": [{"age":60.000000,"anaemia":0.000000,"creatinine_phosphokinase":250.000000,"diabetes":0.000000,"ejection_fraction":38.000000,"high_blood_pressure":0.000000,"platelets":262000.000000,"serum_creatinine":1.10000,"serum_sodium":137.000000,"sex":1.000000,"smoking":0.00000,"time":115.000000}]}

the values can be changed as per desire. After creating the data test sample as mentioned above, I ran the following code:

td = json.dumps(data)

headers = {'Content-Type': 'application/json'}

resp = requests.post(aci_service.scoring_uri, td, headers=headers)    ### sending request to test the deployed webservice

print(resp.json())                                                    ### printing the result of the request sent

the predicted result for the above given data test sample was '0' meaning that the patient did not die during the follow-up period.

Screenshots of deployed model:

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/22.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/23.PNG)

![alt text](https://github.com/ujjwalbb30/nd00333-capstone/blob/master/screenshots/24.PNG)

## Screen Recording

I created the screencast video with the help of Adobe Premier Pro and tried to cover as much steps as possible. It was pretty hard to cover the steps of both 'hyperdrive run' and 'automl run' along with deployment and testing of the best model found, in 5 minutes duration but once I was done with the video I realized that showing maximum amount of information in minimum amount of time is a skill as well.
I uploaded the video on youtube and the link for my screencast video is given below:

https://youtu.be/FQXWJmpqmU0

## Standout Suggestions

I successfully completed one of the standout suggestions.

I converted my best automl model to ONNX format. In order to perform this task, I had to first enable the saving of ONNX compatible model by setting the 'enable_onnx_compatible_models' option to 'True' in 'automl_config'. The code is given below:

automl_config = AutoMLConfig(compute_target=compute_target,
                            task="classification",
                            training_data=dataset,
                            label_column_name="DEATH_EVENT",
                            path='./pipeline-project',
                            enable_early_stopping=True,
                            featurization='auto',
                            enable_onnx_compatible_models=True,
                            debug_log="automl_errors.log",
                            * * automl_settings
)

After the automl run was completed and it predicted VotingEnsemble model as the best model with metric score of 0.9196, I retrieved the model in ONNX format by setting 'return_onnx_model' as 'True' in '.get_output' method of 'remote_run' and saved this model afterwards. The code is given below : 

best_auto_run, best_onnx_model = remote_run.get_output(return_onnx_model=True)

from azureml.automl.runtime.onnx_convert import OnnxConverter           ### importing required dependencies

onnx_fl_path = "./best_model.onnx"                                      ### saving the best model as onnx_model

OnnxConverter.save_onnx_model(best_onnx_model, onnx_fl_path)

## Attribution

I would like to mention the sources I was able to get the help from, to complete this insightful project:

(1) https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml

(2) https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits

(3) https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py

(4) https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#primary-metric

(5) https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb

(6) https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml

(7) https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azclihttps://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/tensorflow/train-hyperparameter-tune-deploy-with-tensorflow/train-hyperparameter-tune-deploy-with-tensorflow.ipynb
