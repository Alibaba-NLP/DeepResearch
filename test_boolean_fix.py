import os

def str_to_bool(value):
    """Convert string to boolean"""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "on")

# Test the original problematic behavior
print("=== Testing Original Behavior ===")
USE_IDP_OLD = os.getenv("USE_IDP", True)
print(f"USE_IDP_OLD type: {type(USE_IDP_OLD)}")
print(f"USE_IDP_OLD value: {USE_IDP_OLD}")
print(f"bool(USE_IDP_OLD): {bool(USE_IDP_OLD)}")

# Test our fixed behavior  
print("\n=== Testing Fixed Behavior ===")
USE_IDP_NEW = str_to_bool(os.getenv("USE_IDP", "True"))
print(f"USE_IDP_NEW type: {type(USE_IDP_NEW)}")
print(f"USE_IDP_NEW value: {USE_IDP_NEW}")
print(f"bool(USE_IDP_NEW): {bool(USE_IDP_NEW)}")

