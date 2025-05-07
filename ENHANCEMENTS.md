# Enhancements

This fork of [1rgs/claude-code-openai](https://github.com/1rgs/claude-code-openai) includes the following improvements:

## 1. Added Support for Gemini 2.5 Pro Preview 05-06

Added `gemini-2.5-pro-preview-05-06` to the list of supported Gemini models, enabling use of this newer model with Claude Code.

```python
# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-05-06", # Added new model
    "gemini-2.0-flash"
]
```

## 2. Fixed Handling of "think" Command

Added special handling for the "think" command to ensure thinking mode works properly with both OpenAI and Gemini models:

```python
@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    # Special handling for "think" command
    if (len(request.messages) == 1 and 
        request.messages[0].role == "user" and
        isinstance(request.messages[0].content, str) and
        request.messages[0].content.strip().lower() == "think"):
        # Set thinking config properly
        request.thinking = ThinkingConfig(enabled=True)
```

## 3. Improved Error Handling for Response Objects

Fixed JSON serialization errors by properly handling Response objects that aren't JSON serializable:

```python
# Handle 'response' attribute specially
if hasattr(e, 'response'):
    response_attr = getattr(e, 'response')
    error_details['response'] = str(response_attr)
```

## Usage

To use these enhancements, configure your `.env` file as follows:

```
# Required API Keys
ANTHROPIC_API_KEY="" # Needed if proxying *to* Anthropic
OPENAI_API_KEY="your_openai_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"

# Use Google as preferred provider
PREFERRED_PROVIDER="google"

# Use Gemini 2.5 Pro Preview (May) for sonnet 
BIG_MODEL="gemini-2.5-pro-preview-05-06"
SMALL_MODEL="gpt-4.1"
```

Then start the server and connect Claude Code to it:

```bash
# Start the server
uv run -m uvicorn server:app --host 0.0.0.0 --port 8082

# In another terminal
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```