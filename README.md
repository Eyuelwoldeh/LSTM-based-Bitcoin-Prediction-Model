# LSTM-based Bitcoin Price Prediction Model

This project implements a **Long Short-Term Memory (LSTM)** model to predict **Bitcoin prices** based on historical data. The model uses a sequence of past Bitcoin prices to forecast future values, leveraging the power of deep learning to capture complex patterns in the time series data.

## Overview

The goal of this project is to build a predictive model that can forecast Bitcoin prices using an LSTM neural network. Time series forecasting tasks, such as predicting stock prices or cryptocurrency values, are well-suited for RNN-based models like LSTM due to their ability to capture long-term dependencies in sequential data.

The model is trained on historical Bitcoin data, and the architecture is optimized for price prediction. The model is evaluated on both training and test datasets and visualized to assess performance.

## Project Structure

- **Data Preprocessing**: The raw data is cleaned, normalized, and formatted into a suitable structure for LSTM input.
- **Model Architecture**: An LSTM-based model is built with one LSTM layer followed by a Dense layer for output.
- **Model Training**: The model is trained over 200 epochs and evaluated on training and validation sets.
- **Model Evaluation**: The performance of the model is evaluated using Root Mean Squared Error (RMSE) and visualized for both training and test predictions.

## Requirements

This project requires the following Python libraries:

- **TensorFlow**: For building and training the LSTM model.
- **Pandas**: For data manipulation and processing.
- **NumPy**: For numerical operations and matrix manipulations.
- **Matplotlib**: For plotting training curves and results.
- **yfinance**: For downloading Bitcoin historical data.
- **scikit-learn**: For data scaling (e.g., MinMaxScaler).

You can install the dependencies by running:

```bash
pip install tensorflow pandas numpy matplotlib yfinance scikit-learn
```

## Data

The dataset used in this project contains Bitcoin's historical market data, specifically the **Open** price. This data is collected in minute-level intervals and spans multiple years.

The data is preprocessed as follows:

- Dates are parsed and converted to the datetime format.
- Non-numeric characters (such as commas, percentage signs, and dollar signs) are removed from the "Open" column.
- The "Open" column is converted to numeric data and missing values are handled.
- The data is normalized using the `MinMaxScaler` to ensure that all values fall within the range of 0 and 1, which helps improve model performance.

## Model Architecture

The model is a simple LSTM network that consists of the following layers:

- **Input Layer**: Takes in the sequence of past Bitcoin prices.
- **LSTM Layer**: The core of the model, designed to capture long-term dependencies in the sequence of prices.
- **Dense Layer**: A fully connected layer that outputs the predicted price.

The model uses **Mean Squared Error (MSE)** as the loss function and **Root Mean Squared Error (RMSE)** as the evaluation metric.

## Training the Model

The model is trained using the training dataset (60% of the data) for **200 epochs**. During training, a checkpoint is saved if the model achieves the best validation performance.

## Model Evaluation

The model's predictions are compared to actual Bitcoin prices for both the training and test sets. The predictions are scaled back to the original range using the inverse transformation of the `MinMaxScaler`. Below are the evaluation results:

### Training vs Test Results

- **Training Loss**: The model performed well on the training set with a loss value steadily decreasing over epochs.
- **Validation Loss**: The model's validation loss also decreased, but with minor fluctuations, indicating good generalization.

### Visualizations

The following plots visualize the model's training and prediction results:

- **Training vs Test Loss**: Shows the loss curve for both training and validation datasets.
- **Predicted vs Actual**: Plots comparing the predicted Bitcoin prices against the actual values.

## Limitations

While this LSTM-based Bitcoin price prediction model demonstrates promising results, there are several limitations that should be taken into account:

### 1. **Data Quality and Preprocessing**:
   - **Missing Data**: The model assumes that all missing data has been handled, but in reality, there may be gaps or errors in the historical Bitcoin price data that could affect predictions. This could lead to inaccuracies if critical periods of market volatility are not represented well in the dataset.
   - **Feature Selection**: The model uses only the "Open" price for predictions. However, factors such as trading volume, closing price, market sentiment, and external news events might have significant influence on price movements and could improve model performance if incorporated.

### 2. **Model Overfitting**:
   - Even though the LSTM model performs well on the training and validation data, there is always the risk of overfitting in time series models, especially when the model architecture is relatively simple or when training on a limited dataset. If the model captures too much noise in the data, it may fail to generalize to new, unseen data, especially during times of extreme market volatility.
   - Overfitting could be further reduced by regularization techniques or by expanding the model architecture to include more layers or units, but this could also increase computation time and complexity.

### 3. **Limited Time Horizon**:
   - The model is trained on historical Bitcoin price data up to a certain point, and it predicts future values based solely on past price patterns. Cryptocurrency markets are highly volatile and can be influenced by factors that the model does not consider, such as regulatory changes, new technological developments, or social media trends. The model may not be able to predict market shifts caused by such events.
   - The model's prediction horizon is limited to short-term price forecasting, and it is not designed for long-term predictions where external factors have a larger impact.

### 4. **Model Complexity and Interpretability**:
   - While LSTMs are effective at modeling sequential data, they can be complex and difficult to interpret. Understanding the exact factors driving the model’s predictions can be challenging. It may not provide any insights into why certain price movements occurred, making it less useful for decision-making in contexts where interpretability is important.
   - Adding explainability features, such as SHAP or LIME, could provide more transparency into how the model is making predictions, but these techniques have limitations when applied to deep learning models like LSTMs.

### 5. **Evaluation Metrics**:
   - The model's performance is evaluated using **Root Mean Squared Error (RMSE)**, which is a common metric for regression tasks. However, RMSE might not always fully reflect the practical usability of the model, especially in financial markets. A model with lower RMSE may still generate predictions that are far from the actual values during periods of high volatility, which is particularly important for cryptocurrency.
   - Other evaluation metrics, such as **Mean Absolute Percentage Error (MAPE)** or **Profit and Loss simulation**, could be used to assess how well the model would perform in a real-world trading scenario.

### 6. **Market Efficiency**:
   - The efficient market hypothesis suggests that asset prices reflect all available information, meaning that past price movements may not always be predictive of future price movements. This implies that the model might have limited predictive power in real-world trading applications, especially in markets with low predictability or during times of high uncertainty.
   - A more advanced approach, such as incorporating additional sources of data (e.g., sentiment analysis from social media or macroeconomic indicators), might improve prediction accuracy in certain cases.

### 7. **Scalability**:
   - The current model is designed for relatively small datasets and may not scale well to handle large volumes of real-time cryptocurrency data. As the model's complexity increases, the training time and memory requirements could grow significantly, potentially limiting its practical use for high-frequency trading or real-time price prediction.
   - Optimizing the model's performance to handle larger datasets and faster prediction times would be necessary for scaling this project to real-time applications.

## Future Improvements

To overcome the limitations mentioned above and improve the accuracy and robustness of the model, future work could involve:

1. **Including Additional Features**: Incorporating other relevant data sources such as trading volume, market sentiment, technical indicators, and external factors (e.g., news or social media sentiment).
2. **Enhancing Model Architecture**: Exploring more complex architectures, such as **GRU** (Gated Recurrent Units) or **Attention Mechanisms**, to capture long-term dependencies more effectively.
3. **Implementing Regularization Techniques**: Utilizing dropout, early stopping, or L2 regularization to reduce overfitting and improve generalization.
4. **Incorporating External Events**: Enhancing the model by integrating sentiment analysis or external financial data to account for events that drive significant market changes.
5. **Longer Prediction Horizon**: Adapting the model for longer-term predictions or developing a hybrid model that incorporates both short-term and long-term factors.
6. **Real-time Trading Simulation**: Testing the model in a real-time trading environment to assess its practical viability, including evaluating its ability to make profitable decisions in volatile market conditions.

## Conclusion

This project demonstrates the potential of LSTM-based models for predicting Bitcoin prices based on historical data. While the model achieves decent results in terms of forecasting short-term price movements, several challenges remain, including data quality, model interpretability, and market unpredictability. Future enhancements to the model can improve its performance and scalability, making it a more robust tool for cryptocurrency price prediction.
