# Human Activity Recognition
This notebook is also available on my Kaggle account: https://www.kaggle.com/code/asmahwimli/human-activity-recognition/notebook
---
This project focuses on recognizing and classifying different physical activities using both raw sensor data and preprocessed features. It highlights the trade-offs between end-to-end learning from raw signals and approaches that rely on domain expertise through handcrafted feature extraction.

## Dataset
The **Human Activity Recognition Using Smartphones (HAR)** dataset was developed to classify human activities using data collected from a smartphone's accelerometer and gyroscope. This data provides insights into body motion patterns during activities like walking, sitting, and lying down.

The experiments involved **30 volunteers** aged 19-48 years. Each volunteer performed six activities:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

**Data Distribution**

<img width="859" height="547" alt="Image" src="https://github.com/user-attachments/assets/a709e514-bd2e-4984-a1f5-81c2d05e7803" />

**Device and Data Collection**:
- Each participant wore a **Samsung Galaxy S II smartphone** on their waist.
- Data was recorded at **50 Hz**(means the sensor collects 50 samples per second) using the smartphone's:
  - **Accelerometer**: Measures 3D linear acceleration (including gravity).
  - **Gyroscope**: Measures 3D angular velocity (rotational motion).
- Data was segmented into sliding windows of **2.56 seconds** with 50% overlap, resulting in **128 readings per window**.

<img width="1490" height="1789" alt="Image" src="https://github.com/user-attachments/assets/eb10b505-cce6-42d3-a3de-d58fc9deb65f" />

Dataset link: https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset

---

## Goal of the project

1. **Raw Sensor Data**:
   - We'll first work with the raw signals from the accelerometer and gyroscope.
   - This will involve processing the **Inertial Signals** files (`body_acc_*` and `body_gyro_*`).
      We will be using an RNN based on GRU units raw data.
2. **Processed Features**:
   - Next, we'll use the precomputed 561-dimensional feature vectors (`X_train.txt` and `X_test.txt`).
   - Fully connected neural networks (FCNs) will classify activities based on these features.

By comparing these two approaches, we aim to understand the strengths and trade-offs of working with raw vs. processed data.

---

## 1. Using raw sensory data
### Model Architecture
The architecture is designed to efficiently process the **sequential sensor data** and predict the activity based on patterns observed in the accelerometer and gyroscope signals.
~~~python
model= Sequential([
    Input((X_train.shape[1],X_train.shape[2])),
    GRU(128, return_sequences=True),
    Dropout(0.1),
    GRU(64, return_sequences=True),
    Dropout(0.1),
    GRU(32, return_sequences=False),
    Dropout(0.1),
    Dense(32, activation="relu"),
    Dense(10, activation="relu"),
    Dense(6, activation="softmax")
])
~~~

### Model Training
**Visualization of loss by epoch**

<img width="576" height="455" alt="Image" src="https://github.com/user-attachments/assets/d9b16d45-a86d-4af7-ab51-d07325fb2ac0" />

As you can see, the model is training well, with both loss steadily decreasing and accuracy increasing over the epochs. This indicates that the model is successfully learning to recognize the activity patterns from the accelerometer and gyroscope data.

---

## 2. Using preprocessed features

### Model Architecture 
The fully connected neural network (FCN) architecture designed for this task consists of several dense layers, which help in classifying the activity based on the extracted features from the accelerometer and gyroscope data. This model does not use any sequential layers like RNNs or GRUs, and instead processes the feature vectors directly through fully connected layers.
~~~python
model_2=Sequential([
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(64, activation="relu"),
    Dense(6, activation="sigmoid")
])
~~~

### Model training
**Visualization of loss by epoch**

<img width="576" height="455" alt="Image" src="https://github.com/user-attachments/assets/ab435549-f810-4ffe-9438-30b7f4635d01" />

The training process for the FCN appears to be more stable, with less noise compared to the GRU-based model that was trained on raw sensor data.

---

## Evaluating and comparing the two models

<img width="469" height="70" alt="Image" src="https://github.com/user-attachments/assets/9d95e292-207b-4801-81c2-80a84c8cf967" />

Overall, the **Fully Connected model** performs better than the **GRU model** in all key metrics—**accuracy**, **precision**, **recall**, and **F1-score**—demonstrating its effectiveness in classifying human activities.

**Effectiveness of the Fully Connected Network**
The **superior performance** of the **Fully Connected model** can largely be attributed to the fact that **the data has been preprocessed and feature-extracted by professionals.** This means that the raw sensor signals from the accelerometer and gyroscope were carefully transformed into **high-quality, normalized features that capture the most relevant information for human activity recognition.** These features are already tailored for classification, allowing the model to directly learn from the meaningful representations of the data, which **significantly improves performance.**

On the other hand, the **GRU model** processes the raw, sequential sensor data without this prior feature extraction, which makes it harder for the model to efficiently learn the patterns needed for classification. While the **GRU model can capture temporal dependencies, it has to learn from less refined data, which increases the complexity and makes it less effective compared to the Fully Connected model that benefits from the feature extraction work done by experts.**

This professional preprocessing step ensures that the Fully Connected model has an advantage, as it operates on data that is already optimized for classification, leading to better accuracy, precision, recall, and F1-score.
