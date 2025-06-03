import json
import uuid
import time
import re
import logging
from typing import List, Dict, Any, Optional, Union, Literal

# Import Pydantic models from models.py
from models import (
    MessagesRequest, MessagesResponse, Usage, ContentBlockText, ContentBlockImage,
    ContentBlockToolUse, ContentBlockToolResult, SystemContent, Tool, ThinkingConfig
)

# Import litellm
import litellm

# Use globally configured logger
# import logging # logging is already imported at the top
logger = logging.getLogger(__name__)

def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    messages = []

    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"

            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                text_content = ""
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            result_content = parse_tool_result_content(getattr(block, "content", ""))
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                             # LiteLLM expects image data in a specific format for OpenAI
                            if anthropic_request.model.startswith("openai/"):
                                processed_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{block.source.get('media_type')};base64,{block.source.get('data')}"}
                                })
                            else: # For other providers, pass as is or adapt as needed
                                processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            processed_content.append({
                                "type": "tool_use", # This might need adjustment based on LiteLLM's expectation for tool use in input
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # This case for assistant messages with tool_result needs careful handling
                            # as LiteLLM/OpenAI expect tool results from 'tool' role, not 'assistant'
                            # For now, we'll represent it as text within the assistant message
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            result_text = parse_tool_result_content(getattr(block, "content", ""))
                            processed_content.append({"type": "text", "text": f"[Tool Result for {tool_id}: {result_text}]"})

                messages.append({"role": msg.role, "content": processed_content})

    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")

    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")
        for tool_config in anthropic_request.tools:
            tool_dict = tool_config.dict()
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema
                }
            }
            openai_tools.append(openai_tool)
        litellm_request["tools"] = openai_tools

    if anthropic_request.tool_choice:
        tool_choice_dict = anthropic_request.tool_choice
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any": # Equivalent to 'required' in OpenAI
            litellm_request["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            litellm_request["tool_choice"] = "auto" # Default

    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any],
                                 original_request: MessagesRequest) -> MessagesResponse:
    try:
        is_claude_model = original_request.model.startswith("anthropic/claude-")

        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        content = []
        if content_text is not None and content_text != "":
            content.append(ContentBlockText(type="text", text=content_text))

        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls for Claude: {tool_calls}")
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments_str = function.get("arguments", "{}")
                else: # Assuming it's an object
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments_str = getattr(function, "arguments", "{}") if function else "{}"

                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments as JSON: {arguments_str}")
                    arguments = {"raw_arguments": arguments_str} # Send as raw string if not parsable

                content.append(ContentBlockToolUse(type="tool_use", id=tool_id, name=name, input=arguments))
        elif tool_calls and not is_claude_model:
            logger.debug(f"Converting tool calls to text for non-Claude model: {original_request.model}")
            tool_text = "\n\nTool usage:\n"
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                tool_text += f"Tool: {name}\nArguments: {arguments}\n\n"

            if content and content[0].type == "text":
                content[0].text += tool_text
            else:
                content.insert(0, ContentBlockText(type="text", text=tool_text))

        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence", # Or map as appropriate
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        if not content: # Ensure content is never empty
            content.append(ContentBlockText(type="text", text=""))

        return MessagesResponse(
            id=response_id,
            model=original_request.model, # Use original model from request
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting LiteLLM response to Anthropic: {str(e)}\n\nTraceback:\n{error_traceback}"
        logger.error(error_message)
        # Fallback response
        return MessagesResponse(
            id=f"error_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[ContentBlockText(type="text", text=f"Error processing LLM response: {str(e)}")],
            stop_reason="end_turn", # Changed "error" to "end_turn" as "error" is not a valid Literal
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""

    # Helper to create a simplified chunk object from JSON string if needed
    class SimpleChunk:
        def __init__(self):
            self.choices = []
            self.usage = None
            self.id = None # For Gemini
            self.model = None # For Gemini

    class SimpleChoice:
        def __init__(self):
            self.finish_reason = None
            self.delta = type('obj', (object,), {"content": None, "tool_calls": None, "role": None}) # role for Gemini
            self.index = 0 # For Gemini

    async def safe_generator_wrapper(generator):
        async for chunk_data in generator:
            if isinstance(chunk_data, str):
                try:
                    chunk_json = json.loads(chunk_data)
                    # This is a simplified parser for stringified JSON chunks
                    # It might need to be more robust depending on LiteLLM's string format
                    simple_chunk = SimpleChunk()
                    simple_choice = SimpleChoice()

                    if "id" in chunk_json: simple_chunk.id = chunk_json["id"]
                    if "model" in chunk_json: simple_chunk.model = chunk_json["model"]

                    delta = {}
                    if "content" in chunk_json: delta["content"] = chunk_json["content"]
                    if "tool_calls" in chunk_json: delta["tool_calls"] = chunk_json["tool_calls"] # Assuming structure
                    if "role" in chunk_json: delta["role"] = chunk_json["role"] # For Gemini

                    if delta: simple_choice.delta = type('obj', (object,), delta)

                    if "finish_reason" in chunk_json: simple_choice.finish_reason = chunk_json["finish_reason"]
                    if "index" in chunk_json: simple_choice.index = chunk_json["index"] # For Gemini

                    simple_chunk.choices.append(simple_choice)

                    if "usage" in chunk_json and chunk_json["usage"] is not None: # For Gemini completion chunk
                         simple_chunk.usage = type('obj', (object,), {
                            "prompt_tokens": chunk_json["usage"].get("prompt_tokens",0), # Gemini uses prompt_token_count
                            "completion_tokens": chunk_json["usage"].get("completion_tokens",0) # Gemini uses candidates_token_count
                        })
                         if "prompt_token_count" in chunk_json["usage"]: # Override if Gemini specific fields are present
                            simple_chunk.usage.prompt_tokens = chunk_json["usage"]["prompt_token_count"]
                         if "candidates_token_count" in chunk_json["usage"]:
                            simple_chunk.usage.completion_tokens = chunk_json["usage"]["candidates_token_count"]


                    yield simple_chunk
                except json.JSONDecodeError:
                    logger.warning(f"Skipping unparseable JSON string chunk: {chunk_data[:100]}...")
                    continue # Skip malformed JSON strings
            else:
                yield chunk_data # Pass through non-string chunks (hopefully LiteLLM ModelResponse objects)

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    # Send initial ping (Anthropic does this)
    yield f"event: ping\ndata: {json.dumps({'type': 'ping', 'timestamp': datetime.utcnow().isoformat()})}\n\n"


    content_block_index = 0
    current_text_block_open = False
    accumulated_output_tokens = 0

    async for chunk in safe_generator_wrapper(response_generator):
        try:
            if not hasattr(chunk, 'choices') or not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage is not None: # Handle Gemini's final usage chunk
                    final_usage_data = {
                        'input_tokens': getattr(chunk.usage, "prompt_tokens",0),
                        'output_tokens': getattr(chunk.usage, "completion_tokens", accumulated_output_tokens) # Use accumulated if available
                    }
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': final_usage_data})}\n\n"
                continue

            choice = chunk.choices[0]
            delta = getattr(choice, 'delta', None)
            if not delta: continue

            delta_content = getattr(delta, 'content', None)
            delta_tool_calls = getattr(delta, 'tool_calls', None)

            # Input tokens are usually sent once at the beginning if available
            # For now, we'll send input_tokens with the first message_delta that has usage info
            # or at the end. Anthropic sends it in message_start but LiteLLM might not provide it early.

            if delta_content:
                if not current_text_block_open:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    current_text_block_open = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                accumulated_output_tokens += litellm.token_counter(model=original_request.model, text=delta_content)


            if delta_tool_calls:
                if current_text_block_open: # Close current text block if open
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                    current_text_block_open = False
                    content_block_index +=1

                for tool_call_delta in delta_tool_calls:
                    tool_call_index = getattr(tool_call_delta, 'index', 0) # OpenAI specific
                    tool_id = getattr(tool_call_delta, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                    function_delta = getattr(tool_call_delta, 'function', None)
                    if not function_delta: continue

                    name = getattr(function_delta, 'name', None)
                    arguments_chunk = getattr(function_delta, 'arguments', None)

                    # Determine effective index for Anthropic content_block
                    # This might need adjustment if multiple tool_calls are streamed piece by piece
                    anthropic_tool_block_index = content_block_index + tool_call_index

                    if name: # Start of a new tool_call
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"

                    if arguments_chunk:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': arguments_chunk}})}\n\n"
                        # Token counting for tool arguments is complex, approximating
                        accumulated_output_tokens += litellm.token_counter(model=original_request.model, text=arguments_chunk)


            finish_reason = getattr(choice, 'finish_reason', None)
            if finish_reason:
                if current_text_block_open:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                    current_text_block_open = False

                # Close any open tool blocks based on their indices
                # This part needs robust tracking of open tool blocks if multiple are handled
                # For simplicity, assuming tools are handled sequentially or one by one for now.
                if delta_tool_calls: # If the finish reason came with tool calls
                    for i, tc_delta in enumerate(delta_tool_calls):
                         # Potentially use tc_delta.index if consistently provided and maps to anthropic_tool_block_index
                         # This assumes the last tool block was at (content_block_index + number of tool calls -1)
                         # A more robust way would be to track open tool_block indices.
                        effective_tool_idx = content_block_index + i
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': effective_tool_idx})}\n\n"


                stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "stop_sequence"}
                anthropic_stop_reason = stop_reason_map.get(finish_reason, "end_turn")

                # Get usage from chunk if available (e.g. from LiteLLM's ModelResponse object)
                usage_data = {'output_tokens': accumulated_output_tokens}
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_data['input_tokens'] = getattr(chunk.usage, 'prompt_tokens', 0)
                    # Override output_tokens if final count is available
                    usage_data['output_tokens'] = getattr(chunk.usage, 'completion_tokens', accumulated_output_tokens)


                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': anthropic_stop_reason}, 'usage': usage_data})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                yield "data: [DONE]\n\n" # Ensure [DONE] is sent for stop
                return

        except Exception as e:
            logger.error(f"Error processing stream chunk: {str(e)}. Chunk: {str(chunk)[:200]}")
            import traceback
            logger.error(traceback.format_exc())
            continue # Try to process next chunk

    # If loop finishes without a finish_reason (e.g. generator ends abruptly)
    if current_text_block_open:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
    # Add similar logic for any open tool blocks if necessary

    # Send a final message_delta and message_stop if not already sent
    # This handles cases where the stream might end without a proper finish_reason from LiteLLM
    # We use the accumulated output tokens for usage. Input tokens might be unknown here.
    final_usage_data = {'output_tokens': accumulated_output_tokens}
    # Try to get input_tokens if the last chunk had it (it might, from some providers)
    if 'chunk' in locals() and hasattr(chunk, 'usage') and chunk.usage and hasattr(chunk.usage, 'prompt_tokens'):
        final_usage_data['input_tokens'] = chunk.usage.prompt_tokens

    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': final_usage_data})}\n\n" # Default to end_turn
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    yield "data: [DONE]\n\n" # Ensure [DONE] is sent

# Need to add `datetime` for `handle_streaming`'s ping
from datetime import datetime
