import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import pandas as pd
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config)
from py_nillion_client import NodeKey, UserKey
from sklearn.linear_model import LinearRegression
from nada_ai.client import SklearnClient

from nillion_utils import compute, store_program, store_secrets, get_user_id_by_seed #local helper file

# Load environment variables from a .env file
load_dotenv()

# Set pandas option to retain old downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Transform Housing.csv dataset to integers
og_housing_data = pd.read_csv('./Housing.csv')
# yes/no to 1/0
og_housing_data = og_housing_data.replace({'yes': 1, 'no': 0}).infer_objects(copy=False)
# furnishingstatus to 2/1/0
og_housing_data['furnishingstatus'] = og_housing_data['furnishingstatus'].replace({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}).infer_objects(copy=False)
# Save the transformed data
housing_data_file = './Housing-transformed.csv'
og_housing_data.to_csv(housing_data_file, index=False)

async def main():
    # Set log scale to match the precision set in the nada program
    na.set_log_scale(32)
    program_name = "linear_regression_12"
    program_mir_path = f"./target/{program_name}.nada.bin"

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")

    # Create 2 parties - Party0 and Party1
    party_names = na_client.parties(2)

    # Create NillionClient for Party0, storer of the model
    seed_0 = 'seed-party-model'
    userkey_party_0 = nillion.UserKey.from_seed(seed_0)
    nodekey_party_0 = nillion.NodeKey.from_seed(seed_0)
    client_0 = create_nillion_client(userkey_party_0, nodekey_party_0)
    party_id_0 = client_0.party_id
    user_id_0 = client_0.user_id

    # Create NillionClient for Party1
    seed_1 = 'seed-party-input'
    userkey_party_1 = nillion.UserKey.from_seed(seed_1)
    nodekey_party_1 = nillion.NodeKey.from_seed(seed_1)
    client_1 = create_nillion_client(userkey_party_1, nodekey_party_1)
    party_id_1 = client_1.party_id
    user_id_1 = client_1.user_id

    # Configure payments
    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    # Party0 stores the linear regression Nada program
    program_id = await store_program(
        client_0,
        payments_wallet,
        payments_client,
        user_id_0,
        cluster_id,
        program_name,
        program_mir_path,
    )

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
    model_secrets = nillion.NadaValues(model_client.export_state_as_secrets("my_model", na.SecretRational))

    # create permissions for model_secrets: Party0 has default permissions, Party1 has compute permissions
    permissions_for_model_secrets = nillion.Permissions.default_for_user(user_id_0)

    # along with user_id_1, allow other user ids to use the secret in the linear regression program by specifying user key seeds
    allowed_user_ids = [user_id_1, get_user_id_by_seed("inference_1"), get_user_id_by_seed("inference_2"), get_user_id_by_seed("inference_3")]
    permissions_dict = {user: {program_id} for user in allowed_user_ids}
    permissions_for_model_secrets.add_compute_permissions(permissions_dict)

    # Party0 stores the model as a Nillion Secret
    model_store_id = await store_secrets(
        client_0,
        payments_wallet,
        payments_client,
        cluster_id,
        model_secrets,
        1,
        permissions_for_model_secrets,
    )

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
    input_secrets = nillion.NadaValues(my_input)

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party(party_names[0], party_id_0)
    compute_bindings.add_input_party(party_names[1], party_id_1)
    compute_bindings.add_output_party(party_names[1], party_id_1)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {model_store_id}")

    # Party1 performs blind compuptation that runs inference and returns the result
    # Compute, passing all params including the receipt that shows proof of payment
    inference_result = await compute(
        client_1,
        payments_wallet,
        payments_client,
        program_id,
        cluster_id,
        compute_bindings,
        [model_store_id],
        input_secrets,
        verbose=True,
    )
    
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
