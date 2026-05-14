# Predicting DDOS Attacks using a LSTM Time Series model
The notebook is also available on my kaggle account: https://www.kaggle.com/code/asmahwimli/predicting-ddosattacks-with-lstm-time-series-model
## DDos Attacks:
A distributed denial-of-service (DDoS) attack is a malicious attempt to disrupt the normal traffic of a targeted server, service or network by overwhelming the target or its surrounding infrastructure with a flood of Internet traffic.

From a high level, a DDoS attack is like an unexpected traffic jam clogging up the highway, preventing regular traffic from arriving at its destination.

![ddos_attack_traffic_metaphor](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/96a3d42b-ea7d-421c-b04c-11cbded12b1e)

## How to identify a DDoS attack:
The most obvious symptom of a DDoS attack is a site or service suddenly becoming slow or unavailable. But since a number of causes — such a legitimate spike in traffic — can create similar performance issues, further investigation is usually required. Traffic analytics tools can help you spot some of these telltale signs of a DDoS attack:

* Suspicious amounts of traffic originating from a single IP address or IP range
* A flood of traffic from users who share a single behavioral profile, such as device type, geolocation, or web browser version
* An unexplained surge in requests to a single page or endpoint
* Odd traffic patterns such as spikes at odd hours of the day or patterns that appear to be unnatural (e.g. a spike every 10 minutes)
* There are other, more specific signs of DDoS attack that can vary depending on the type of attack.

## Project goal: Utilize LSTM-based time series modeling to predict DDoS attacks with the aim of enhancing early detection and proactive mitigation strategies.

## Dataset: 
The used training and validation data is available on kaggle: https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019

Which is part of the CICDDoS2017 dataset, which is an academic dataset collected by the Canadian Institute for Cybersecurity using CICFlowMeterV3, specifically dedicated to detecting distributed denial of service (DDoS) attacks. This data was collected on July 7, 2017, between 3:30 and 5:30.

### Distribution of label : Benign or DDOS in the dataset: 
![distribution_of_ddos_attacks](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/4c7d96f7-7e02-45d3-b60b-caa4304cb09a)

The distribution of target variable ('Label') in the dataset reflects a well-balanced representation of the two classes, with approximately 43.3% classified as benign and 56.7% as DDoS attacks, indicating a balanced dataset composition.

## Data preprocessing:
In this phase, I did some data preprocessing steps including:
* Dropping constant columns
* Correlation analysis: Since we have 84 columns, This is such an enormous correlation matrix, it will be hard from this to identify highly correlated features. I identified a threshold (absolute correlation > 0.9) to identify highly correlated features.
* Feature Selection: Next, I proceeded to drop highly correlated columns to address multicollinearity issues and reduce redundancy in the dataset. I performed this step iteratively, considering different sets of features and their correlations with the target variable ('Label'). By carefully evaluating the correlation between each feature and the target variable, I aimed to retain the most informative and independent features while eliminating those that might introduce noise or redundancy into the model. This iterative process helped me refine the feature set and improve the predictive performance of the model.
* Setting the time column `Timestamp` as in index: Since we will be working on a Time series model, the Timestamp column won't be considered as a feature but an index.
* Label Encoding categorical features
* Scaling numerical data: using a Z-score
## Preparing the data for the time series model:
During this step, I'm setting up the data for training the model, specifically for a time series analysis task.
I initialized two empty lists trainX and trainY to store the input sequences and corresponding target values for our model training.
* n_future = 1: This variable represents how many time steps in the future we aim to predict.
* n_past = 10: This variable denotes the number of past time steps our model will utilize to make predictions about the future.
For Loop:
* We slice the DataFrame to extract the past n_past rows as input features (trainX), representing the historical data that our model will use for prediction.
* We select the next row after n_future time steps as the target variable (trainY), which represents the label or value to be predicted by our model.

![code](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/d0ce266c-141e-4ef6-9ba2-1edd2f6f41f4)

![code2](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/06d17c85-cb9c-41e7-9940-f2b55c554cb2)

*The shapes of our training data arrays are as follows: trainX has a shape of (225735, 10, 34), indicating that we have 225,735 samples, each consisting of a sequence of 10 time steps, with 34 features for each time step. For each sample, the model will receive the past 10 observations, each containing information about various features.

*On the other hand, trainY has a shape of (225735, 1), meaning that for each of these samples, the model will aim to predict a single binary outcome: whether the subsequent observation represents a DDOS attack or not. This setup allows the model to learn patterns in the historical data to make predictions about future DDOS attacks.

## The LSTM model:
I built a sequential model using LSTM neural networks for predicting DDoS attacks. The model structure is:
* First LSTM layer: 128 units, returns complete sequences to capture dependencies at various time scales.
* Second LSTM layer: 64 units, also returns complete sequences to capture complex temporal dependencies.
* Third LSTM layer: 32 units, returns only the last output state, summarizing the sequence information into a single vector.
* Dropout layer: Regularization with a dropout rate of 0.01 to prevent overfitting.
* Dense layer: A single unit with sigmoid activation, typical for binary classification to predict the presence of a DDoS attack.
![model_summary](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/f6109ac4-f57b-4d4a-be4c-8216021382c4)


## Training:
![loss_by_epochs](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/ae169893-84b7-4140-8f6e-387a39b9d354)
![accuracy_by_epochs](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/d59acd68-ddb0-467b-9ff0-f4d541f9069c)


## Model evaluation: 
I tried to evaluate the data on another dataset available here: https://github.com/neonithinar/Intrusion-detection-cicDDOS-dataset/tree/main

This dataset presents different types of DDOS attacks, which will allow me to test how well the model is identifying these types as ddos attacks.
![distribution_of_ddos_attacks2](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/6308bab2-e197-4a9d-aa12-8380f78daea3)

Overall, it seems like the model achieved good performance with high accuracy, precision, recall, and F1 score on the test set, indicating that it's able to effectively detect various types of attacks.


![evaluation](https://github.com/Asma1233670/Predicting_ddos_attacks_using_time_series_model/assets/75503007/bf24c8f4-fb0e-4039-917e-ce3acc2a8728)



