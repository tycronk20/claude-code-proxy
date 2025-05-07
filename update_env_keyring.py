#!/usr/bin/env python3
"""
Script to retrieve API keys from macOS keychain and update .env file.
Uses the keyring library which is cross-platform and more secure.
"""

import os
import sys
from dotenv import load_dotenv

# Define keychain service names
SERVICES = {
    "ANTHROPIC_API_KEY": "anthropic-api-key",
    "OPENAI_API_KEY": "openai-api-key",
    "GEMINI_API_KEY": "gemini-api-key"
}

# Path to .env file
ENV_FILE = ".env"

def create_env_file():
    """Create .env file from example if it doesn't exist"""
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as example, open(ENV_FILE, "w") as env:
            env.write(example.read())
        print(f"Created {ENV_FILE} from .env.example")
    else:
        # Create minimal .env file
        with open(ENV_FILE, "w") as env:
            env.write("""# Required API Keys
ANTHROPIC_API_KEY=""
OPENAI_API_KEY=""
GEMINI_API_KEY=""

# Optional: Provider Preference and Model Mapping
PREFERRED_PROVIDER="openai"
BIG_MODEL="gpt-4.1"
SMALL_MODEL="gpt-4.1-mini"
""")
        print(f"Created minimal {ENV_FILE} file")

def update_env_file():
    """Update .env file with keys from keychain"""
    import keyring
    
    # Create .env file if it doesn't exist
    if not os.path.exists(ENV_FILE):
        create_env_file()
    
    # Load existing .env file
    load_dotenv(ENV_FILE)
    env_vars = {}
    comments = {}
    
    # Read existing .env file into dictionaries
    with open(ENV_FILE, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                # Check if there's a comment at the end of the line
                if "#" in value:
                    val_part, comment_part = value.split("#", 1)
                    env_vars[key] = val_part.strip()
                    comments[key] = f" #{comment_part}"
                else:
                    env_vars[key] = value.strip()
                    comments[key] = ""
    
    # Get keys from keychain
    updated = False
    username = os.environ.get("USER", "default")
    
    for env_var, service in SERVICES.items():
        print(f"Attempting to get {env_var} from keychain service '{service}'...")
        try:
            # Get password from keychain
            api_key = keyring.get_password(service, username)
            
            if api_key:
                print(f"Found key for {env_var} in keychain.")
                # Update our env_vars dictionary with the new value
                env_vars[env_var] = f'"{api_key}"'
                updated = True
            else:
                print(f"Key for {env_var} not found in keychain.")
        except Exception as e:
            print(f"Error retrieving {env_var}: {e}")
    
    # Write updated env vars back to file
    if updated:
        # Preserve comments and formatting
        with open(ENV_FILE, "r") as file:
            lines = file.readlines()
        
        with open(ENV_FILE, "w") as file:
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in env_vars:
                        # Write the updated line with any comment preserved
                        file.write(f"{key}={env_vars[key]}{comments.get(key, '')}\n")
                        # Remove key from dict so we know it's been written
                        env_vars.pop(key, None)
                        if key in comments:
                            comments.pop(key)
                    else:
                        file.write(line)
                else:
                    file.write(line)
            
            # Add any new keys that weren't in the original file
            for key, value in env_vars.items():
                file.write(f"{key}={value}{comments.get(key, '')}\n")
        
        print(f"Updated {ENV_FILE} with API keys from keychain.")
    else:
        print("No keys were updated. Check your keychain service names.")

if __name__ == "__main__":
    # Check if keyring is installed
    try:
        import keyring
    except ImportError:
        print("Error: keyring package not installed.")
        print("Install it with: pip install keyring python-dotenv")
        sys.exit(1)
    
    update_env_file()