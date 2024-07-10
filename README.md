# NBA Player Performance ML Project

## Overview

This project leverages machine learning techniques to predict NBA player performance and rank players based on their performance metrics across three seasons (2021-2024). By utilizing datasets from Kaggle and various machine learning libraries in Python, the project aims to provide insights into player efficiency, team dynamics, and performance trends.

## Motivation

In the competitive landscape of the NBA, data-driven insights can significantly influence team strategies and player appraisals. This project focuses on utilizing machine learning to:
- Assess player performance beyond traditional statistics.
- Identify undervalued players and potential stars.
- Inform scouting, signings, and game strategies.
- Predict performance declines due to physical stress to prevent long-term damage to players.

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation
- **Numpy**: Numerical computations
- **Matplotlib**: Data visualization
- **TensorFlow**: Machine learning framework

## Project Structure

### Data Collection and Preprocessing

1. **Data Loading**: Reading CSV files for each season's player statistics.
2. **Data Merging**: Combining data from three seasons into a single dataset.
3. **Normalization**: Applying z-score normalization to selected features.

### Neural Network Model

The model is a Fully Connected Neural Network (FCNN) designed for regression to predict normalized ranks of NBA players.

- **Input Layer**: Processes input features.
- **Hidden Layers**: Multiple layers with ReLU activation, L2 regularization, and dropout to prevent overfitting.
- **Output Layer**: A single neuron outputs the predicted rank.

#### Configuration
- **Layer 1**: 128 neurons, ReLU activation, L2 regularization
- **Dropout**: 50% dropout rate
- **Layer 2**: 64 neurons, similar configuration
- **Dropout**: 50% dropout rate
- **Layer 3**: 32 neurons, similar configuration
- **Dropout**: 50% dropout rate
- **Output Layer**: Single neuron

#### Optimization and Loss Function
- **Optimizer**: Adam optimizer
- **Loss Function**: Mean Squared Error (MSE)

### Training

The model is trained for 100 epochs with a batch size of 10, using a 70-30 train-test split to ensure sufficient data for learning and validation.

### Performance Evaluation

Performance is evaluated based on:
- **Loss and Validation Loss**
- **Mean Absolute Error (MAE) and Validation MAE**
- **Mean Squared Error (MSE) and Validation MSE**

## Results and Conclusions

The model demonstrated consistent improvement in predicting player rankings, as indicated by decreasing loss and error rates over 100 epochs. This suggests the effectiveness of the chosen approach and model configuration. 

Challenges included the time lag in obtaining new NBA season data for immediate testing and transforming the model into a more accessible format. Future plans involve creating a simulation environment and collaborating with software development experts to enhance the model's practical application.
