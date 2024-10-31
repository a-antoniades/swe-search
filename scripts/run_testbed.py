import logging

from testbed.sdk.sdk import TestbedSDK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logging.getLogger().setLevel(logging.INFO)
def run_sdk():
    sdk = TestbedSDK()
    client = sdk.create_client("django__django-13230")
    print(client.status())
    client.wait_until_ready()

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)
    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)

    response = client.run_evaluation()
    print(response)
    client.destroy()


run_sdk()