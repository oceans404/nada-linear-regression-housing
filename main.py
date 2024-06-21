import os
import asyncio
import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from dotenv import load_dotenv
from nillion_python_helpers import create_nillion_client, getNodeKeyFromFile, getUserKeyFromFile
from sklearn.linear_model import LinearRegression
from nada_ai.client import SklearnClient
import pandas as pd

from nillion_uitls import compute, store_program, store_secrets

# Load environment variables from a .env file
load_dotenv()

# transform housing data set to integers
housing_data = pd.read_csv('./Housing.csv')
# yes/no to 1/0
housing_data.replace({'yes': 1, 'no': 0}, inplace=True)
# furnishingstatus to 2/1/0
housing_data['furnishingstatus'].replace({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}, inplace=True)
# save the transformed data
housing_data.to_csv('./Housing-transformed.csv', index=False)

async def main():
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    seed = 'test-linear-regression'
    userkey = nillion.UserKey.from_seed(seed)
    nodekey = nillion.NodeKey.from_seed(seed)
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id
    party_names = na_client.parties(2)
    program_name = "linear_regression_12"
    program_mir_path = f"./target/{program_name}.nada.bin"

    # Store the program
    program_id = await store_program(client, user_id, cluster_id, program_name, program_mir_path)

    # Load the housing dataset
    data = pd.read_csv('./Housing-transformed.csv')
    features = data.columns.tolist()
    features.remove('price')
    X = data[features].values  # All housing features except 'price'
    y = data['price'].values  # Target price

    # Train a linear regression
    model = LinearRegression()
    fit_model = model.fit(X, y)

    print("Learned model coeffs are:", model.coef_)
    print("Learned model intercept is:", model.intercept_)

    # Create and store model secrets via ModelClient
    model_client = SklearnClient(fit_model)
    model_secrets = nillion.Secrets(model_client.export_state_as_secrets("my_model", na.SecretRational))

    model_store_id = await store_secrets(client, cluster_id, program_id, party_id, party_names[0], model_secrets)

    # home features
    area=2000
    bedrooms=3
    bathrooms=2
    stories=1
    mainroad=1
    guestroom=0
    basement=1
    hotwaterheating=1
    airconditioning=1
    parking=2
    prefarea=0
    furnishingstatus=0

    dream_home = [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]
    new_house = np.array(dream_home)
    my_input = na_client.array(new_house, "my_input", na.SecretRational)
    input_secrets = nillion.Secrets(my_input)

    data_store_id = await store_secrets(client, cluster_id, program_id, party_id, party_names[1], input_secrets)

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)
    [
        compute_bindings.add_input_party(party_name, party_id)
        for party_name in party_names
    ]
    compute_bindings.add_output_party(party_names[1], party_id)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {model_store_id} {data_store_id}")

    # Perform the computation and return the result
    result = await compute(client, cluster_id, compute_bindings, [model_store_id, data_store_id], nillion.Secrets({}))
    
    # Rescale the obtained result by the quantization scale
    outputs = [na_client.float_from_rational(result["my_output"])]
    # Convert the result from its rational form to a floating-point number
    print(f"🖥️  The rescaled result is {outputs}")

    expected = fit_model.predict(new_house.reshape(1, -1))
    print(f"""
    Features:
        house area: {area}
        # bedrooms: {bedrooms}
        # bathrooms: {bathrooms}
        # stories: {stories}
        is connected to the mainroad: {'yes' if mainroad == 1 else 'no'}
        has guestroom: {'yes' if guestroom == 1 else 'no'}
        has basement: {'yes' if basement == 1 else 'no'}
        has hotwaterheating: {'yes' if hotwaterheating == 1 else 'no'}
        has airconditioning" {'yes' if airconditioning == 1 else 'no'}
        # parking spots: {parking}
        has prefarea: {'yes' if prefarea == 1 else 'no'}
        is furnished: {'unfurnished' if furnishingstatus == 0 else 'semi-furnished' if furnishingstatus == 1 else'furnished'}
    """)
    print(f"The predicted price of this home is ${"{:,.2f}".format(expected[0])}")
    return result

# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
