from dotenv import load_dotenv

from testbed.sdk import TestbedSDK

load_dotenv()
testbed_sdk = TestbedSDK()
testbed_sdk.cleanup_user_resources()
