# Linear regression on housing data with blind inference

Inference is the process of using a trained model to make predictions or decisions based on new, unseen data. It is the phase where the model is applied to input data to generate an output, such as a prediction or classification.

In this example I ran blind inference on [Nillion](https://docs.nillion.com) to predict the house price of a new input home based on 12 features: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus. **The inference is "blind" because the party running the price prediction with the new input never sees the trained model state. Instead, the trained model state is provided to the Nada program as a secret.**

Result of running [main.py](./main.py):

- <img width="670" alt="Screenshot 2024-06-24 at 10 23 45 AM" src="https://github.com/oceans404/nada-linear-regression-housing/assets/91382964/5920575d-d2ba-4805-99b2-dd303261eeda">

Model trained with this housing dataset: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data

## How it works

Two parties, Party0 and Party1, are involved in the blind inference process. The Nada program linear_regression_12.py is designed to handle 12 specific housing features. 

Party0's Role:
  - Party0 stores the linear_regression_12 Nada program in the network.
  - Party0 takes the housing dataset and trains a linear regression model using scikit-learn.
  - After training, Party0 stores the model state as a secret in Nillion, giving Party1 computation permissions for this secret.

Party1's Role:
  - Party1 creates a new input with an array of features (e.g., area, bedrooms, bathrooms, etc.).
  - Party1 computes with the linear_regression_12 program to run blind inference on the secret trained model. This involves passing their new input as a compute-time secret.
  - Party1 receives the output of linear_regression_12, which is the predicted price of a house based on their input features without ever seeing the trained model state, ensuring the model parameters remain confidential.

## Run this example

[Install the nillion sdk](https://docs.nillion.com/nillion-sdk-and-tools#installation)

Run nillion-devnet and populate .env

```
bootstrap.sh
```

Optionally update home feature values in the `main.py` file to change new_input to your desired home traits

Run `python3 main.py`
