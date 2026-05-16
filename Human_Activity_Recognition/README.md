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

