import os
import asyncio
import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from dotenv import load_dotenv
from nillion_python_helpers import create_nillion_client
from sklearn.linear_model import LinearRegression
from nada_ai.client import SklearnClient
import pandas as pd

from nillion_utils import compute, store_program, store_secrets

# Load environment variables from a .env file
load_dotenv()

# transform Housing.csv dataset to integers
og_housing_data = pd.read_csv('./Housing.csv')
# yes/no to 1/0
og_housing_data.replace({'yes': 1, 'no': 0}, inplace=True)
# furnishingstatus to 2/1/0
og_housing_data['furnishingstatus'].replace({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}, inplace=True)
# save the transformed data
housing_data_file = './Housing-transformed.csv'
og_housing_data.to_csv(housing_data_file, index=False)

async def main():
    # Set log scale to match the precision set in the nada program
    na.set_log_scale(32)
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    program_name = "linear_regression_12"
    program_mir_path = f"./target/{program_name}.nada.bin"

    # Create 2 parties - Party0 and Party1
    party_names = na_client.parties(2)

    # Create NillionClient for Party0
    seed_0 = 'seed-party-0'
    userkey_party_0 = nillion.UserKey.from_seed(seed_0)
    nodekey_party_0 = nillion.NodeKey.from_seed(seed_0)
    client_0 = create_nillion_client(userkey_party_0, nodekey_party_0)
    party_id_0 = client_0.party_id
    user_id_0 = client_0.user_id

    # Create NillionClient for Party1
    seed_1 = 'seed-party-1'
    userkey_party_1 = nillion.UserKey.from_seed(seed_1)
    nodekey_party_1 = nillion.NodeKey.from_seed(seed_1)
    client_1 = create_nillion_client(userkey_party_1, nodekey_party_1)
    party_id_1 = client_1.party_id
    user_id_1 = client_1.user_id

    # Party0 stores the linear regression Nada program
    program_id = await store_program(client_0, user_id_0, cluster_id, program_name, program_mir_path)

    # Load the transformed housing dataset
    data = pd.read_csv(housing_data_file)
    features = data.columns.tolist()
    features.remove('price')
    # X is all housing features except price
    X = data[features].values  
    # y target price
    y = data['price'].values

    # Train a linear regression with sklearn model
    model = LinearRegression()
    fit_model = model.fit(X, y)

    print("Learned model coeffs are:", model.coef_)
    print("Learned model intercept is:", model.intercept_)

    # Create SklearnClient with nada-ai
    model_client = SklearnClient(fit_model)

    # Party0 creates a secret
    model_secrets = nillion.Secrets(model_client.export_state_as_secrets("my_model", na.SecretRational))

    # create permissions for model_secrets: Party0 has default permissions, Party1 has compute permissions
    permissions_for_model_secrets = nillion.Permissions.default_for_user(user_id_0)
    permissions_for_model_secrets.add_compute_permissions({
        user_id_1: {program_id},
    })

    # Party0 stores the model as a Nillion Secret
    model_store_id = await store_secrets(client_0, cluster_id, program_id, party_id_0, party_names[0], model_secrets, permissions_for_model_secrets)

    # Party1 creates the new input secret, which will be provided to compute as a compute time secret
    # home features
    area=3000
    bedrooms=4
    bathrooms=3
    stories=2
    mainroad=0
    guestroom=1
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

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party(party_names[0], party_id_0)
    compute_bindings.add_input_party(party_names[1], party_id_1)
    compute_bindings.add_output_party(party_names[1], party_id_1)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {model_store_id}")

    # Party1 performs blind compuptation that runs inference and return the result
    inference_result = await compute(client_1, cluster_id, compute_bindings, [model_store_id], input_secrets)
    
    # Rescale the obtained result by the quantization scale
    outputs = [na_client.float_from_rational(inference_result["my_output"])]
    # Convert the result from its rational form to a floating-point number
    print(f"🙈 The rescaled result computed by the {program_name} Nada program is {outputs[0]}")
    expected = fit_model.predict(new_house.reshape(1, -1))
    print(f"👀  The expected result computed by sklearn is {expected[0]}")

    print(f"""
    Features of new input home:
        house area: {area}
        # bedrooms: {bedrooms}
        # bathrooms: {bathrooms}
        # stories: {stories}
        is connected to the mainroad: {'yes' if mainroad == 1 else 'no'}
        has guestroom: {'yes' if guestroom == 1 else 'no'}
        has basement: {'yes' if basement == 1 else 'no'}
        has hotwaterheating: {'yes' if hotwaterheating == 1 else 'no'}
        has airconditioning: {'yes' if airconditioning == 1 else 'no'}
        # parking spots: {parking}
        has prefarea: {'yes' if prefarea == 1 else 'no'}
        is furnished: {'unfurnished' if furnishingstatus == 0 else 'semi-furnished' if furnishingstatus == 1 else'furnished'}
    """)
    print(f"The predicted price of this home is ${"{:,.2f}".format(outputs[0])}")
    return inference_result

# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
