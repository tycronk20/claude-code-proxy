import logging
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal

# Import config variables
from config import (
    PREFERRED_PROVIDER, BIG_MODEL, SMALL_MODEL,
    OPENAI_MODELS, GEMINI_MODELS
)

# Use globally configured logger
import logging
logger = logging.getLogger(__name__)


from pydantic import model_validator # Import model_validator

# Standalone function for model validation and mapping logic
def _map_model_name_logic(input_model_name: str, existing_original_model: Optional[str]) -> tuple[str, str]:
    """
    Maps a model name based on predefined rules and returns the mapped name and the original name.
    If existing_original_model is provided, it's used; otherwise, input_model_name is considered original.
    """
    original_to_log = existing_original_model or input_model_name
    new_model = input_model_name # Start with the input model name

    logger.debug(f"📋 MODEL MAPPING: Input='{input_model_name}', Original='{original_to_log}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

    clean_v = new_model # Use new_model for cleaning, not input_model_name directly after this point
    if clean_v.startswith('anthropic/'):
        clean_v = clean_v[10:]
    elif clean_v.startswith('openai/'):
        clean_v = clean_v[7:]
    elif clean_v.startswith('gemini/'):
        clean_v = clean_v[7:]

    mapped = False
    if 'haiku' in clean_v.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{SMALL_MODEL}"
            mapped = True
        else:
            new_model = f"openai/{SMALL_MODEL}"
            mapped = True
    elif 'sonnet' in clean_v.lower():
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{BIG_MODEL}"
            mapped = True
        else:
            new_model = f"openai/{BIG_MODEL}"
            mapped = True
    elif not mapped: # Only apply prefix if no specific mapping occurred
        if clean_v in GEMINI_MODELS and not new_model.startswith('gemini/'):
            new_model = f"gemini/{clean_v}"
            mapped = True
        elif clean_v in OPENAI_MODELS and not new_model.startswith('openai/'):
            new_model = f"openai/{clean_v}"
            mapped = True

    final_original_model = existing_original_model or input_model_name

    if mapped: # Log only if a change (mapping or prefixing) actually happened
        logger.debug(f"📌 MODEL MAPPING RESULT: Input='{input_model_name}' (Original='{final_original_model}') ➡️ Mapped='{new_model}'")
    elif not new_model.startswith(('openai/', 'gemini/', 'anthropic/')):
         logger.warning(f"⚠️ No prefix or mapping rule for model: '{new_model}' (Original='{final_original_model}'). Using as is.")

    return new_model, final_original_model


# Standalone function for model validation and mapping logic
# This function is now simpler, primarily for use by field_validator if complex logic was needed there.
# However, with model_validator, the main mapping logic is better placed there or in a helper like above.
def _validate_and_map_model_name(v: str, info: Any, context: str = "MODEL") -> str:
    # This function's role changes. It will just log and ensure `original_model` is in info.data
    # The actual mapping logic will be called by a model_validator.
    # For now, this function can simply store the initial 'v' as 'original_model' in info.data
    # if it's not already there, and return 'v'. The model_validator will handle the mapping.

    # If original_model is already set (e.g. by direct init), don't overwrite with just 'v'
    # This function becomes mostly a pass-through for the field_validator context,
    # as the main logic is shifted to the model_validator.
    # It could still be used for initial validation if needed, but mapping is deferred.

    if hasattr(info, 'data') and isinstance(info.data, dict) and 'original_model' not in info.data:
         # Store the initial model name from the request if original_model isn't explicitly provided.
         # This ensures that if someone passes `model="claude-3-haiku"` and `original_model="claude-3-haiku-custom"`
         # the latter is respected by the model_validator.
         # If `original_model` is not provided at all, then `v` (the input to model field) is the original.
        info.data['original_model_from_field'] = v # Temporary store

    # The field validator for 'model' should now primarily validate the type of 'v' if necessary,
    # but the mapping itself is better handled by a model_validator.
    # For simplicity, we'll let the model_validator handle everything.
    # This function can effectively become a no-op or basic type check if desired.
    # For now, just return 'v'.
    return v

class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = False
    budget_tokens: Optional[int] = None
    type: Optional[str] = None

    @classmethod
    def enabled_config(cls):
        """Create a default enabled thinking config"""
        return cls(enabled=True)

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('thinking', mode='before') # Changed to 'before'
    @classmethod # Ensure it's a classmethod
    def validate_thinking(cls, v):
        if isinstance(v, dict) and v.get('type') == 'enabled' and 'enabled' not in v:
            # budget_tokens is optional in ThinkingConfig, so allow its absence for conversion
            budget_tokens = v.get('budget_tokens')
            type_val = v.get('type') # type is also optional in ThinkingConfig but required by this custom dict format
            logger.debug(f"Converting thinking dict for MessagesRequest to ThinkingConfig: {v}")
            return ThinkingConfig(enabled=True, budget_tokens=budget_tokens, type=type_val)
        # If v is already a ThinkingConfig, or None, or a dict not matching the specific structure,
        # Pydantic will handle it (e.g., coerce dict to ThinkingConfig using defaults, or pass through None/ThinkingConfig)
        return v

    @field_validator('model')
    def validate_model_field_basic(cls, v, info):
        # This field validator now does minimal validation or passes through.
        # It can store the initial 'v' if original_model is not provided in __init__.
        if hasattr(info, 'data') and isinstance(info.data, dict):
            if 'original_model' not in info.data or info.data['original_model'] is None:
                 # If original_model wasn't given at init, this 'model' field's input is the original.
                info.data['original_model'] = v
        return v

    @model_validator(mode='after')
    def process_model_mapping(self) -> 'MessagesRequest':
        mapped_name, original_name = _map_model_name_logic(self.model, self.original_model)
        self.model = mapped_name
        self.original_model = original_name # Ensure original_model on self is correctly set
        return self

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator('thinking', mode='before') # Changed to 'before' for TokenCountRequest
    @classmethod # Ensure it's a classmethod
    def validate_thinking(cls, v): # Validator for TokenCountRequest.thinking
        if isinstance(v, dict) and v.get('type') == 'enabled' and 'enabled' not in v:
            budget_tokens = v.get('budget_tokens')
            type_val = v.get('type')
            logger.debug(f"Converting thinking dict for TokenCountRequest to ThinkingConfig: {v}")
            return ThinkingConfig(enabled=True, budget_tokens=budget_tokens, type=type_val)
        return v

    @field_validator('model')
    def validate_model_field_basic_token_count(cls, v, info):
        # Similar to MessagesRequest, store initial 'v' if original_model not set
        if hasattr(info, 'data') and isinstance(info.data, dict):
            if 'original_model' not in info.data or info.data['original_model'] is None:
                info.data['original_model'] = v
        return v

    @model_validator(mode='after')
    def process_model_mapping_token_count(self) -> 'TokenCountRequest':
        mapped_name, original_name = _map_model_name_logic(self.model, self.original_model)
        self.model = mapped_name
        self.original_model = original_name
        return self

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
