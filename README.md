# Machine-learning-Time-series-models


# 🌍 ARIMA Forecasting for SDG Index Score 🚀

Welcome to the **ARIMA Forecasting** project, where we leverage **time series analysis** to forecast the **Sustainable Development Goals (SDG) Index Scores** for **all countries around the world**. This project uses the powerful ARIMA (AutoRegressive Integrated Moving Average) model to analyze trends, patterns, and predict the future trajectory of SDG scores across different countries. Ready to dive into the world of data-driven sustainability? Let's get started! 🌱
![image](https://github.com/user-attachments/assets/c6a234c8-a109-49ba-ba91-de0718f929f1)

To cite my paper you can use this link :  
Chenary, K., Pirian Kalat, O., & Sharifi, A. (2024). Forecasting Sustainable Development Goals Scores by 2030 Using Machine Learning Models. Sustainable Development. https://doi.org/10.1002/sd.3037

## 🔧 Requirements

To run this script, you'll need to install the following Python libraries:

- `numpy`: For mathematical operations.
- `pandas`: For data manipulation.
- `matplotlib`: For visualizations.
- `seaborn`: For advanced plotting.
- `statsmodels`: For statistical models, including ARIMA.
- `pmdarima`: For automatic ARIMA parameter selection.
- `sklearn`: For machine learning tools.
- `google.colab`: For uploading files if using Google Colab.

You can easily install the required libraries via pip:

```bash
pip install numpy pandas matplotlib seaborn statsmodels pmdarima sklearn
```

## 📊 Dataset

This project utilizes SDG Index Score data, specifically for **OECD countries**. The data spans multiple years and serves as the foundation for our forecasts. It is contained within the CSV file **`fullsdg_final2023_witoutmissing.csv`**.

### Data Columns:

- **year**: The year corresponding to the SDG Index Score.
- **SDG Index Score**: The score that represents a country's progress toward sustainable development goals.

### Benchmarks:
This project benchmarks the performance of various machine learning models on small time series datasets, specifically for predicting SDG scores (Sustainable Development Goals scores) using global regional groups as predictors. The benchmark includes key performance metrics such as execution time and mean squared error (MSE) for each model.

![image](https://github.com/user-attachments/assets/9b2e4127-f627-4eff-9a5a-da79f61b8f1f)


## 🛠️ Key Steps in the Code

### 1. Data Preprocessing:

The dataset is cleaned and filtered to focus on **OECD countries**. We convert the `year` column into a **datetime format** and set it as the index. The data is then split into **training** and **testing** datasets based on a cutoff date (January 1, 2016).

```python
df = pd.read_csv('fullsdg_final2023_witoutmissing.csv')
df = df[df['Country'] == 'OECD members']
df = df[['year', 'SDG Index Score']]
df['year'] = pd.to_datetime(df['year'], format='%Y').dt.strftime('%Y-01-01')
df['year'] = pd.to_datetime(df['year'])
```

### 2. 📈 Data Visualization:

Here's a powerful time series plot that visualizes the SDG Index Score over time for both the training and testing datasets:

![Time Series Plot](images/train_test_plot.png)

### 3. 📉 Stationarity Check:

To make sure our time series is **stationary** (a requirement for ARIMA), we perform the **Augmented Dickey-Fuller (ADF)** test. If needed, the data is differenced to achieve stationarity.

### 4. 🔍 ACF and PACF Plots:

The **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots are essential to determine the optimal parameters for the ARIMA model.

![image](https://github.com/user-attachments/assets/abc96e75-2942-40d4-ba91-23f5f3dc1322)


### 5. 🔧 ARIMA Model Fitting:

We fit the ARIMA model on the stationary data and evaluate it using the **Mean Squared Error (MSE)**. Predictions are made using this model for forecasting future values.
![image](https://github.com/user-attachments/assets/ed465a88-4772-459e-aa4e-3a5a48640ba3)


### 6. 📅 Prediction & Forecasting:

With the ARIMA model in place, we predict the SDG Index Scores for future years, from **2016 to 2040**. Here’s the forecast plot with **confidence intervals** that give us an understanding of prediction uncertainty.


![image](https://github.com/user-attachments/assets/83239a95-9e35-47fc-87d8-ba6c3db2e556)
![image](https://github.com/user-attachments/assets/b3b88cea-0160-49e5-87f4-273a2134d9a3)
![image](https://github.com/user-attachments/assets/880e50bf-3d80-4f8b-b74a-f4f2ddca5f3f)


## ⚡ Notes

- **Don’t forget**: Make sure you upload the correct dataset or adjust the file path if working locally.
- **Tuning is key**: The ARIMA model is tuned using the **auto_arima** function, which makes it easy to find the best configuration automatically.
- **Model evaluation**: The model is evaluated using **Mean Squared Error (MSE)**, ensuring its prediction quality.


