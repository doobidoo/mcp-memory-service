import toml

# Path to the uv.lock file
file_path = "c:/REPOSITORIES/mcp-memory-service/uv.lock"

# Load the TOML data
with open(file_path, "r") as file:
    uv_lock_data = toml.load(file)

# Extract package information
packages = uv_lock_data.get("package", [])
requirements = [
    f"{pkg['name']}=={pkg['version']}" for pkg in packages
]

# Print the requirements
print("\n".join(requirements))
