# Keychain Integration for API Keys

This guide explains how to store your API keys in macOS Keychain and use them with the server.

## Setting Up Keychain for API Keys

1. Open "Keychain Access" app (in Applications > Utilities or search with Spotlight)

2. For each API key, create a new password item:
   - Click "+" button or go to File > New Password Item
   - Fill in the following details:
     - **Service/Name**: Use these exact names:
       - `anthropic-api-key` for Anthropic
       - `openai-api-key` for OpenAI
       - `gemini-api-key` for Gemini/Google
     - **Account Name**: Your macOS username
     - **Password**: Your actual API key

3. Repeat for each API key you want to store

## Using Python Keyring to Access Keys

The `update_env_keyring.py` script will extract your API keys from keychain and update the .env file:

1. Install required packages:
   ```
   pip install keyring python-dotenv
   ```

2. Run the script:
   ```
   ./update_env_keyring.py
   ```

The script will:
- Create .env file if it doesn't exist
- Retrieve API keys from your keychain
- Update the .env file while preserving other settings

## How It Works

The script uses Python's `keyring` library to securely access the macOS keychain:

```python
import keyring

# Get API key from keychain
api_key = keyring.get_password("service-name", "account-name")
```

This approach is:
- More secure than hardcoding keys
- Cross-platform (works on macOS, Windows, Linux)
- Easy to automate for regular updates

## Troubleshooting

If keys aren't found:
- Verify service names match exactly: `anthropic-api-key`, `openai-api-key`, `gemini-api-key`
- Check your account name matches your macOS username
- Ensure keychain is unlocked when running the script