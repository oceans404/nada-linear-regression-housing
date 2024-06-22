# Linear regression on housing data with blind inference

Predict housing prices based on 12 features: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus

Housing dataset: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data

The Nada program linear_regression_12.py is hardocded with the feature count (12 for the number of housing features of interest) for the dataset. Party0 stores the `linear_regression_12` Nada program.

Party0 takes the housing dataset and uses it to train a linear model with scikit-learn. Then Party0 stores the model state in Nillion as a Secret.

Party1 creates a new input with an array of features.

Party1 computes `linear_regression_12` to run blind inference on the secret trained data, passing in their new input as a compute time secret. Party1 gets the expected price of a house with their input features.

## Run this example

[Install the nillion sdk](https://docs.nillion.com/nillion-sdk-and-tools#installation)

Run nillion-devnet and populate .env

```
bootstrap.sh
```

Optionally update home feature values in the `main.py` file to change new_input to your desired home traits

Run `python3 main.py`
