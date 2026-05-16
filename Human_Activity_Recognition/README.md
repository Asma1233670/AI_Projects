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

 
  
    

    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }


  
    
      
      Accuracy
      Precision
      Recall
      F1-Score
    
  
  
    
      GRU Model on Raw Data
      0.909060
      0.913080
      0.909060
      0.909152
    
    
      Fully Connected Model on Preprocessed Data
      0.936206
      0.939576
      0.936206
      0.936456
    
  


    

  
    

  
    
  
    

  
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  

    
      const buttonEl =
        document.querySelector('#df-91f65583-e1ab-499d-bde4-e17ddc00ce2b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-91f65583-e1ab-499d-bde4-e17ddc00ce2b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    
  


  
    
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    
    

  
    
  
    
    
      (() => {
      const buttonEl =
        document.querySelector('#id_d636f5f7-8379-4fdc-b2e6-d60517ebb9bc button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('metrics_df');
      }
      })();
    
  

    
  

