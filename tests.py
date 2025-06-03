import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import json
import importlib
import logging
from typing import List, Dict, Any, Union, Literal, Optional, Generator # Ensure Generator is imported for async generator typing

# Modules to test
import config
import models
import conversion

# Pydantic models for type hinting and instance creation in tests
from pydantic import BaseModel # Not strictly needed here as models are imported but good for clarity
from models import (
    ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult,
    SystemContent, Message, Tool, ThinkingConfig, MessagesRequest,
    TokenCountRequest, TokenCountResponse, Usage, MessagesResponse
)

# Suppress most logging during tests unless specifically testing logging output
logging.getLogger("config").setLevel(logging.CRITICAL + 1)
logging.getLogger("models").setLevel(logging.CRITICAL + 1)
logging.getLogger("conversion").setLevel(logging.CRITICAL + 1)


class TestConfig(unittest.TestCase):

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key",
                             "OPENAI_API_KEY": "test_openai_key",
                             "GEMINI_API_KEY": "test_gemini_key",
                             "LOG_LEVEL": "INFO"})
    def test_api_keys_and_log_level_loaded(self):
        importlib.reload(config)
        self.assertEqual(config.ANTHROPIC_API_KEY, "test_anthropic_key")
        self.assertEqual(config.OPENAI_API_KEY, "test_openai_key")
        self.assertEqual(config.GEMINI_API_KEY, "test_gemini_key")
        self.assertEqual(config.LOG_LEVEL, "INFO")


    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(config)
            self.assertEqual(config.PREFERRED_PROVIDER, "openai")
            self.assertEqual(config.BIG_MODEL, "gpt-4.1")
            self.assertEqual(config.SMALL_MODEL, "gpt-4.1-mini")
            self.assertEqual(config.LOG_LEVEL, "WARN")

    @patch.dict(os.environ, {"PREFERRED_PROVIDER": "google", "BIG_MODEL": "gemini-ultra",
                             "SMALL_MODEL": "gemini-flash", "LOG_LEVEL": "DEBUG"})
    def test_override_default_values(self):
        importlib.reload(config)
        self.assertEqual(config.PREFERRED_PROVIDER, "google")
        self.assertEqual(config.BIG_MODEL, "gemini-ultra")
        self.assertEqual(config.SMALL_MODEL, "gemini-flash")
        self.assertEqual(config.LOG_LEVEL, "DEBUG")


    def test_model_lists_loaded(self):
        importlib.reload(config)
        self.assertIsInstance(config.OPENAI_MODELS, list)
        self.assertTrue(len(config.OPENAI_MODELS) > 0)
        self.assertIn("gpt-4.1", config.OPENAI_MODELS)

        self.assertIsInstance(config.GEMINI_MODELS, list)
        self.assertTrue(len(config.GEMINI_MODELS) > 0)
        self.assertIn("gemini-2.0-flash", config.GEMINI_MODELS)


class TestModels(unittest.TestCase):

    def setUp(self):
        class MockInfo:
            def __init__(self):
                self.data = {}
        self.mock_info = MockInfo()

    @patch('models.PREFERRED_PROVIDER', 'openai')
    @patch('models.BIG_MODEL', 'gpt-4-large-test')
    @patch('models.SMALL_MODEL', 'gpt-4-small-test')
    @patch('models.GEMINI_MODELS', ['gemini-pro-test', 'gemini-flash-test'])
    @patch('models.OPENAI_MODELS', ['gpt-4-large-test', 'gpt-4-small-test', 'gpt-3.5-turbo-test'])
    def test_map_model_name_logic_openai_preference(self): # Renamed test
        # Test Haiku mapping
        mapped_name, original_name = models._map_model_name_logic("anthropic/claude-3-haiku-20240307", None)
        self.assertEqual(mapped_name, "openai/gpt-4-small-test")
        self.assertEqual(original_name, "anthropic/claude-3-haiku-20240307")

        # Test Sonnet mapping
        mapped_name, original_name = models._map_model_name_logic("anthropic/claude-3-sonnet-20240229", None)
        self.assertEqual(mapped_name, "openai/gpt-4-large-test")
        self.assertEqual(original_name, "anthropic/claude-3-sonnet-20240229")

        # Test OpenAI model prefixing
        mapped_name, original_name = models._map_model_name_logic("gpt-3.5-turbo-test", None)
        self.assertEqual(mapped_name, "openai/gpt-3.5-turbo-test")
        self.assertEqual(original_name, "gpt-3.5-turbo-test")

        # Test Gemini model prefixing
        mapped_name, original_name = models._map_model_name_logic("gemini-flash-test", None)
        self.assertEqual(mapped_name, "gemini/gemini-flash-test")
        self.assertEqual(original_name, "gemini-flash-test")

        # Test unknown model (no prefix, no mapping)
        mapped_name, original_name = models._map_model_name_logic("unknown-model-123", None)
        self.assertEqual(mapped_name, "unknown-model-123")
        self.assertEqual(original_name, "unknown-model-123")

        # Test with existing_original_model provided
        mapped_name, original_name = models._map_model_name_logic("claude-3-haiku", "custom-haiku-original") # input_model_name is different from original
        self.assertEqual(mapped_name, "openai/gpt-4-small-test") # Mapping should apply to input_model_name "claude-3-haiku"
        self.assertEqual(original_name, "custom-haiku-original") # Original should be preserved


    @patch('models.PREFERRED_PROVIDER', 'google')
    @patch('models.BIG_MODEL', 'gemini-pro-test')
    @patch('models.SMALL_MODEL', 'gemini-flash-test')
    @patch('models.GEMINI_MODELS', ['gemini-pro-test', 'gemini-flash-test'])
    @patch('models.OPENAI_MODELS', ['gpt-4-large-test', 'gpt-4-small-test'])
    def test_map_model_name_logic_google_preference(self): # Renamed test
        # Test Haiku mapping to Gemini
        mapped_name, original_name = models._map_model_name_logic("anthropic/claude-3-haiku-20240307", None)
        self.assertEqual(mapped_name, "gemini/gemini-flash-test")
        self.assertEqual(original_name, "anthropic/claude-3-haiku-20240307")

        # Test Sonnet mapping to Gemini
        mapped_name, original_name = models._map_model_name_logic("anthropic/claude-3-sonnet-20240229", None)
        self.assertEqual(mapped_name, "gemini/gemini-pro-test")
        self.assertEqual(original_name, "anthropic/claude-3-sonnet-20240229")

    def test_message_request_validation_integration(self):
        with patch('models.PREFERRED_PROVIDER', 'openai'), \
             patch('models.SMALL_MODEL', 'gpt-4-small-test'), \
             patch('models.OPENAI_MODELS', ['gpt-4-small-test']):

            req = MessagesRequest(
                model="claude-3-haiku-20240307",
                messages=[Message(role="user", content="Hello")],
                max_tokens=10,
                original_model=None # Explicitly include for Pydantic to track
            )
            self.assertEqual(req.model, "openai/gpt-4-small-test")
            self.assertEqual(req.model_dump().get('original_model'), "claude-3-haiku-20240307")

    def test_token_count_request_validation_integration(self):
        with patch('models.PREFERRED_PROVIDER', 'google'), \
             patch('models.SMALL_MODEL', 'gemini-flash-test'), \
             patch('models.GEMINI_MODELS', ['gemini-flash-test']):

            req = TokenCountRequest(
                model="claude-3-haiku-20240307",
                messages=[Message(role="user", content="Hello")],
                original_model=None # Explicitly include for Pydantic to track
            )
            self.assertEqual(req.model, "gemini/gemini-flash-test")
            self.assertEqual(req.model_dump().get('original_model'), "claude-3-haiku-20240307")

    def test_thinking_config_enabled_config(self):
        tc = ThinkingConfig.enabled_config()
        self.assertTrue(tc.enabled)
        self.assertIsNone(tc.budget_tokens)
        self.assertIsNone(tc.type)

    def test_thinking_config_dict_validator(self):
        req = MessagesRequest(
            model="gpt-4", max_tokens=10, messages=[],
            thinking={"budget_tokens": 100, "type": "enabled"} # type: ignore
        )
        self.assertIsInstance(req.thinking, ThinkingConfig)
        self.assertTrue(req.thinking.enabled)
        self.assertEqual(req.thinking.budget_tokens, 100)

        req_token_count = TokenCountRequest(
            model="gpt-4", messages=[],
            thinking={"budget_tokens": 50, "type": "enabled"} # type: ignore
        )
        self.assertIsInstance(req_token_count.thinking, ThinkingConfig)
        self.assertTrue(req_token_count.thinking.enabled)

class TestConversion(unittest.TestCase):

    def test_clean_gemini_schema(self):
        # Test basic removal of 'additionalProperties' and 'default' at root and nested
        schema1 = {
            "type": "object",
            "properties": {
                "prop1": {"type": "string", "default": "value1"},
                "prop2": {"type": "object", "properties": {"nested_prop": {"default": "value2"}}, "default": {}}
            },
            "additionalProperties": False, # This one is fine by Gemini, but often removed by users
            "default": {}
        }
        cleaned1 = conversion.clean_gemini_schema(json.loads(json.dumps(schema1)))
        self.assertNotIn("default", cleaned1) # Root default removed
        self.assertNotIn("default", cleaned1["properties"]["prop1"]) # Nested default in prop1 removed
        self.assertNotIn("default", cleaned1["properties"]["prop2"]) # Nested default for prop2 object removed
        self.assertNotIn("default", cleaned1["properties"]["prop2"]["properties"]["nested_prop"]) # Deeply nested default removed
        # 'additionalProperties': False can be kept if desired, but the function might remove it if it was True
        # The current clean_gemini_schema removes additionalProperties regardless of its boolean value if it's at the root,
        # but not if it's within a property. Let's test that specific behavior.

        schema_add_props = {"type": "object", "additionalProperties": True, "properties": {"p1": {"type":"string", "additionalProperties": False}}}
        cleaned_add_props = conversion.clean_gemini_schema(json.loads(json.dumps(schema_add_props)))
        self.assertNotIn("additionalProperties", cleaned_add_props) # Root additionalProperties removed
        self.assertNotIn("additionalProperties", cleaned_add_props["properties"]["p1"]) # Nested one also removed by current logic

        # Test format removal (unsupported vs supported)
        schema2 = {
            "type": "object",
            "properties": {
                "email_prop": {"type": "string", "format": "email"}, # unsupported
                "hostname_prop": {"type": "string", "format": "hostname"}, # unsupported
                "date_prop": {"type": "string", "format": "date-time"}, # supported
                "enum_prop": {"type": "string", "enum": ["a","b"], "format": "enum"} # supported format "enum"
            }
        }
        cleaned2 = conversion.clean_gemini_schema(json.loads(json.dumps(schema2)))
        self.assertNotIn("format", cleaned2["properties"]["email_prop"])
        self.assertNotIn("format", cleaned2["properties"]["hostname_prop"])
        self.assertIn("format", cleaned2["properties"]["date_prop"])
        self.assertEqual(cleaned2["properties"]["date_prop"]["format"], "date-time")
        self.assertIn("format", cleaned2["properties"]["enum_prop"]) # "enum" format should be kept
        self.assertEqual(cleaned2["properties"]["enum_prop"]["format"], "enum")


        # Test nested schemas within properties (object) and items (array)
        schema3 = {
            "type": "object",
            "properties": {
                "nested_object": {
                    "type": "object",
                    "properties": {"nes_prop1": {"type": "string", "default": "abc", "format": "ipv4"}},
                    "additionalProperties": True # This should be removed from the nested object's definition
                },
                "nested_array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"arr_item_prop": {"type": "string", "format": "uuid", "default":"id"}},
                        "default": {} # This default on the item's schema object should be removed
                    }
                }
            }
        }
        cleaned3 = conversion.clean_gemini_schema(json.loads(json.dumps(schema3)))
        # Check nested object
        self.assertNotIn("additionalProperties", cleaned3["properties"]["nested_object"])
        self.assertNotIn("default", cleaned3["properties"]["nested_object"]["properties"]["nes_prop1"])
        self.assertNotIn("format", cleaned3["properties"]["nested_object"]["properties"]["nes_prop1"])
        # Check nested array items
        self.assertNotIn("default", cleaned3["properties"]["nested_array"]["items"])
        self.assertNotIn("default", cleaned3["properties"]["nested_array"]["items"]["properties"]["arr_item_prop"])
        self.assertNotIn("format", cleaned3["properties"]["nested_array"]["items"]["properties"]["arr_item_prop"])
        
        # Test that it doesn't break a valid schema that doesn't need cleaning much
        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"}
            },
            "required": ["name"]
        }
        cleaned_valid = conversion.clean_gemini_schema(json.loads(json.dumps(valid_schema)))
        self.assertEqual(cleaned_valid, valid_schema, "Schema that required no cleaning was altered.")

    def test_parse_tool_result_content(self):
        self.assertEqual(conversion.parse_tool_result_content("text"), "text")
        self.assertEqual(conversion.parse_tool_result_content(None), "No content provided")
        # Adjusted to reflect that parse_tool_result_content when processing a list of blocks
        # that results in a single line of text, will strip the final implicit newline.
        self.assertEqual(conversion.parse_tool_result_content([{"type": "text", "text": "hi"}]), "hi")
        # If multiple blocks result in multiple lines, newlines between them are preserved, but trailing one might be stripped by final .strip()
        self.assertEqual(conversion.parse_tool_result_content([{"type": "text", "text": "hi"}, {"type": "text", "text": "bye"}]), "hi\nbye")
        # For mixed content, if it ends with a string, strip() applies to the whole result.
        self.assertEqual(conversion.parse_tool_result_content([{"type": "other"}, "str"]), '{"type": "other"}\nstr')
        self.assertEqual(conversion.parse_tool_result_content({"type": "text", "text": "dict_text"}), "dict_text")
        self.assertEqual(conversion.parse_tool_result_content({"a":1}), '{"a": 1}')


    @patch('conversion.clean_gemini_schema', side_effect=lambda x: x)
    @patch('conversion.logger')
    def test_convert_anthropic_to_litellm(self, mock_logger, mock_clean_schema):
        # Basic request with simple user/assistant string messages
        anth_req_simple = MessagesRequest(
            model="claude-3-sonnet-20240229", # Input model name
            max_tokens=100,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?")
            ]
        )
        # anth_req_simple.model will be the mapped name due to Pydantic model validation
        llm_req_simple = conversion.convert_anthropic_to_litellm(anth_req_simple)
        self.assertEqual(llm_req_simple["model"], anth_req_simple.model) # Expect the mapped model name
        self.assertEqual(len(llm_req_simple["messages"]), 3)
        self.assertEqual(llm_req_simple["messages"][0]["role"], "user")
        self.assertEqual(llm_req_simple["messages"][0]["content"], "Hello")
        self.assertEqual(llm_req_simple["messages"][1]["role"], "assistant")
        self.assertEqual(llm_req_simple["messages"][1]["content"], "Hi there!")


        # System message conversion (string and list of SystemContent blocks)
        anth_req_sys_str = MessagesRequest(model="claude-3", max_tokens=10, messages=[], system="You are an assistant.")
        llm_req_sys_str = conversion.convert_anthropic_to_litellm(anth_req_sys_str)
        self.assertEqual(llm_req_sys_str["messages"][0]["role"], "system")
        self.assertEqual(llm_req_sys_str["messages"][0]["content"], "You are an assistant.")

        anth_req_sys_list = MessagesRequest(model="claude-3", max_tokens=10, messages=[], system=[SystemContent(type="text", text="Assistant prompt.")])
        llm_req_sys_list = conversion.convert_anthropic_to_litellm(anth_req_sys_list)
        self.assertEqual(llm_req_sys_list["messages"][0]["role"], "system")
        self.assertEqual(llm_req_sys_list["messages"][0]["content"], "Assistant prompt.")

        # Anthropic tools to LiteLLM/OpenAI format + clean_gemini_schema call for Gemini
        tool_schema = {"type": "object", "properties": {"location": {"type": "string", "default": "SF"}}}
        anth_req_tools_gemini = MessagesRequest(
            model="gemini/gemini-1.5-pro", max_tokens=50, messages=[],
            tools=[Tool(name="get_weather", description="Get weather", input_schema=tool_schema)]
        )
        conversion.convert_anthropic_to_litellm(anth_req_tools_gemini)
        mock_clean_schema.assert_called_with(tool_schema) # Called for Gemini
        
        mock_clean_schema.reset_mock()
        anth_req_tools_claude = MessagesRequest(
            model="claude-3-opus", max_tokens=50, messages=[],
            tools=[Tool(name="get_weather", description="Get weather", input_schema=tool_schema)]
        )
        llm_req_tools_claude = conversion.convert_anthropic_to_litellm(anth_req_tools_claude)
        mock_clean_schema.assert_not_called() # Not called for non-Gemini
        self.assertEqual(llm_req_tools_claude["tools"][0]["type"], "function") # type: ignore
        self.assertEqual(llm_req_tools_claude["tools"][0]["function"]["name"], "get_weather") # type: ignore
        self.assertEqual(llm_req_tools_claude["tools"][0]["function"]["parameters"], tool_schema) # type: ignore


        # Tool choice conversion
        anth_req_tc_tool = MessagesRequest(model="claude-3", max_tokens=10, messages=[], tool_choice={"type":"tool", "name":"my_tool"})
        llm_req_tc_tool = conversion.convert_anthropic_to_litellm(anth_req_tc_tool)
        self.assertEqual(llm_req_tc_tool["tool_choice"]["type"], "function") # type: ignore
        self.assertEqual(llm_req_tc_tool["tool_choice"]["function"]["name"], "my_tool") # type: ignore

        anth_req_tc_any = MessagesRequest(model="claude-3", max_tokens=10, messages=[], tool_choice={"type":"any"})
        llm_req_tc_any = conversion.convert_anthropic_to_litellm(anth_req_tc_any)
        self.assertEqual(llm_req_tc_any["tool_choice"], "required") # "any" maps to "required"

        anth_req_tc_auto = MessagesRequest(model="claude-3", max_tokens=10, messages=[], tool_choice={"type":"auto"})
        llm_req_tc_auto = conversion.convert_anthropic_to_litellm(anth_req_tc_auto)
        self.assertEqual(llm_req_tc_auto["tool_choice"], "auto")


        # User message with multiple ContentBlock types, including tool_result
        user_msg_multi_content = MessagesRequest(
            model="claude-3", max_tokens=150,
            messages=[Message(role="user", content=[
                ContentBlockText(type="text", text="What is the capital of France?"),
                ContentBlockImage(type="image", source={"type": "base64", "media_type": "image/jpeg", "data": "base64_encoded_data"}),
                ContentBlockToolResult(type="tool_result", tool_use_id="tool_search_123", content=[{"type":"text", "text":"Paris is the capital."}])
            ])]
        )
        llm_user_multi_content = conversion.convert_anthropic_to_litellm(user_msg_multi_content)
        # The current implementation merges text and tool_result into a single string message for user role.
        # Image blocks are converted if model is openai, otherwise might be passed differently or text-only.
        final_content_str = llm_user_multi_content["messages"][0]["content"]
        self.assertIn("What is the capital of France?", final_content_str) # type: ignore
        self.assertIn("Tool result for tool_search_123:\nParis is the capital.", final_content_str) # type: ignore
        # For non-OpenAI models, image content might be trickier to assert if it's just passed as a complex object.
        # If model was "openai/..." then image block would be structured differently.
        # Let's test the OpenAI image case specifically:
        user_msg_image_openai = MessagesRequest(
            model="openai/gpt-4o", max_tokens=10,
            messages=[Message(role="user", content=[
                ContentBlockText(type="text", text="Describe:"),
                ContentBlockImage(type="image", source={"type": "base64", "media_type": "image/jpeg", "data": "imgdata"})
            ])]
        )
        llm_img_openai = conversion.convert_anthropic_to_litellm(user_msg_image_openai)
        self.assertIsInstance(llm_img_openai["messages"][0]["content"], list)
        content_list = llm_img_openai["messages"][0]["content"]
        self.assertEqual(content_list[0]["type"], "text") # type: ignore
        self.assertEqual(content_list[0]["text"], "Describe:") # type: ignore
        self.assertEqual(content_list[1]["type"], "image_url") # type: ignore
        self.assertEqual(content_list[1]["image_url"]["url"], "data:image/jpeg;base64,imgdata") # type: ignore


        # Max tokens capping
        anth_req_tokens_openai = MessagesRequest(model="openai/gpt-4o", max_tokens=20000, messages=[])
        llm_req_tokens_openai = conversion.convert_anthropic_to_litellm(anth_req_tokens_openai)
        self.assertEqual(llm_req_tokens_openai["max_tokens"], 16384)

        anth_req_tokens_gemini = MessagesRequest(model="gemini/gemini-pro", max_tokens=19000, messages=[])
        llm_req_tokens_gemini = conversion.convert_anthropic_to_litellm(anth_req_tokens_gemini)
        self.assertEqual(llm_req_tokens_gemini["max_tokens"], 16384)
        
        anth_req_tokens_claude = MessagesRequest(model="claude-3-opus", max_tokens=5000, messages=[])
        llm_req_tokens_claude = conversion.convert_anthropic_to_litellm(anth_req_tokens_claude)
        self.assertEqual(llm_req_tokens_claude["max_tokens"], 5000) # No capping for non-OpenAI/Gemini


    @patch('conversion.logger')
    def test_convert_litellm_to_anthropic(self, mock_logger):
        # Basic LiteLLM dict response
        orig_req_claude = MessagesRequest(model="anthropic/claude-3-opus-20240229", max_tokens=10, messages=[])
        llm_resp_dict = {
            "id": "chatcmpl-123", "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5} # total_tokens is ignored by our func
        }
        anth_resp = conversion.convert_litellm_to_anthropic(llm_resp_dict, orig_req_claude)
        self.assertEqual(anth_resp.id, "chatcmpl-123")
        self.assertEqual(anth_resp.model, "anthropic/claude-3-opus-20240229") # Uses original model
        self.assertEqual(anth_resp.content[0].type, "text")
        self.assertEqual(anth_resp.content[0].text, "Hello!")
        self.assertEqual(anth_resp.stop_reason, "end_turn") # "stop" maps to "end_turn"
        self.assertEqual(anth_resp.usage.input_tokens, 10)
        self.assertEqual(anth_resp.usage.output_tokens, 5)

        # Mocking LiteLLM's ModelResponse object
        # Create a more faithful mock of litellm.ModelResponse if its structure is complex
        # For now, MagicMock with attributes should suffice for what's accessed.
        mock_litellm_model_response = MagicMock(spec=True) # Use spec=True for stricter attribute access
        mock_litellm_model_response.id = "resp_obj_789"
        mock_litellm_model_response.model = "litellm-used-model" # This is the model LiteLLM reports it used
        
        mock_message_obj = MagicMock(spec=True)
        mock_message_obj.role = "assistant"
        mock_message_obj.content = "Response from ModelResponse."
        mock_message_obj.tool_calls = None # No tool calls in this case
        
        mock_choice_obj = MagicMock(spec=True)
        mock_choice_obj.message = mock_message_obj
        mock_choice_obj.finish_reason = "length" # Should map to "max_tokens"
        
        mock_litellm_model_response.choices = [mock_choice_obj]
        
        mock_usage_obj = MagicMock(spec=True)
        mock_usage_obj.prompt_tokens = 15
        mock_usage_obj.completion_tokens = 30
        mock_litellm_model_response.usage = mock_usage_obj
        
        anth_resp_from_obj = conversion.convert_litellm_to_anthropic(mock_litellm_model_response, orig_req_claude)
        self.assertEqual(anth_resp_from_obj.id, "resp_obj_789")
        self.assertEqual(anth_resp_from_obj.model, orig_req_claude.model) # Should take from original request
        self.assertEqual(anth_resp_from_obj.content[0].text, "Response from ModelResponse.") # type: ignore
        self.assertEqual(anth_resp_from_obj.stop_reason, "max_tokens") # "length" maps to "max_tokens"
        self.assertEqual(anth_resp_from_obj.usage.input_tokens, 15)
        self.assertEqual(anth_resp_from_obj.usage.output_tokens, 30)


        # Tool calls for Claude model (original request was for Claude)
        llm_tool_resp_claude_case = {
            "id": "tool_resp_claude",
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [ # content is None for tool_calls
                {"id": "tc1_claude", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'}}
            ]}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10}
        }
        anth_tool_resp_claude = conversion.convert_litellm_to_anthropic(llm_tool_resp_claude_case, orig_req_claude)
        self.assertEqual(len(anth_tool_resp_claude.content), 1) # Should be one tool_use block
        self.assertEqual(anth_tool_resp_claude.content[0].type, "tool_use")
        self.assertEqual(anth_tool_resp_claude.content[0].name, "get_weather") # type: ignore
        self.assertEqual(anth_tool_resp_claude.content[0].input, {"location":"Paris"}) # type: ignore
        self.assertEqual(anth_tool_resp_claude.stop_reason, "tool_use")

        # Tool calls for non-Claude model (e.g. OpenAI model in original request)
        orig_req_openai = MessagesRequest(model="openai/gpt-4o", max_tokens=10, messages=[])
        llm_tool_resp_openai_case = {
            "id": "tool_resp_openai",
            "choices": [{"message": {"role": "assistant", "content": "Okay, using a tool.", "tool_calls": [ # Text can exist alongside tool_calls
                {"id": "tc1_openai", "type": "function", "function": {"name": "search_web", "arguments": '{"query":"Pydantic"}'}}
            ]}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 25, "completion_tokens": 15}
        }
        anth_tool_resp_openai = conversion.convert_litellm_to_anthropic(llm_tool_resp_openai_case, orig_req_openai)
        # For non-Claude, tool_calls are converted to text and appended to existing text content.
        self.assertEqual(len(anth_tool_resp_openai.content), 1) # Should be one text block
        self.assertEqual(anth_tool_resp_openai.content[0].type, "text")
        self.assertIn("Okay, using a tool.", anth_tool_resp_openai.content[0].text) # type: ignore
        self.assertIn("Tool: search_web", anth_tool_resp_openai.content[0].text) # type: ignore
        self.assertIn('Arguments: {"query":"Pydantic"}', anth_tool_resp_openai.content[0].text) # type: ignore
        self.assertEqual(anth_tool_resp_openai.stop_reason, "tool_use")

        # Test empty/None content from LiteLLM
        llm_empty_content_resp = {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens":1, "completion_tokens":0}}
        anth_empty_content = conversion.convert_litellm_to_anthropic(llm_empty_content_resp, orig_req_claude)
        self.assertEqual(len(anth_empty_content.content), 1)
        self.assertEqual(anth_empty_content.content[0].type, "text")
        self.assertEqual(anth_empty_content.content[0].text, "") # Empty string if content was None

        # Test error handling fallback
        mock_orig_req_err = MagicMock(spec=MessagesRequest)
        mock_orig_req_err.model = "mock_model_for_error_test"
            
        problematic_litellm_response = MagicMock(spec=True)
        problematic_litellm_response.id = "litellm_mock_id"

        mock_usage_obj_err = MagicMock(spec=True)
        mock_usage_obj_err.prompt_tokens = 0
        mock_usage_obj_err.completion_tokens = 0
        problematic_litellm_response.usage = mock_usage_obj_err
        
        # Configure .choices to be a property that raises an error when accessed.
        type(problematic_litellm_response).choices = unittest.mock.PropertyMock(side_effect=ValueError("Simulated error on choices access"))
            
        error_anth_resp = conversion.convert_litellm_to_anthropic(problematic_litellm_response, mock_orig_req_err)
            
        self.assertTrue(error_anth_resp.id.startswith("error_"))
        self.assertEqual(error_anth_resp.content[0].type, "text")
        self.assertIn("Error processing LLM response: Simulated error on choices access", error_anth_resp.content[0].text)
        self.assertEqual(error_anth_resp.stop_reason, "end_turn")


import asyncio # Add asyncio import

@patch('conversion.litellm.token_counter', return_value=1)
def test_handle_streaming_events(self, mock_token_counter):
    async def async_test_logic():
        # Mock objects for simulating LiteLLM streaming chunks
        # These need to match the structure LiteLLM's acompletion stream yields
        class MockDelta:
            def __init__(self, content=None, tool_calls=None, role=None):
                self.content = content
                self.tool_calls = tool_calls
                self.role = role
        class MockFunctionDelta:
            def __init__(self, name=None, arguments=None):
                self.name = name
                self.arguments = arguments
        class MockToolCallDelta: # Represents a part of a tool_call in a stream
            def __init__(self, id=None, type="function", index=0, function=None): # id can be None initially
                self.id = id
                self.type = type
                self.index = index # OpenAI uses index for multiple tool calls
                self.function = function
        class MockChoice:
            def __init__(self, delta, finish_reason=None, index=0): # index for choice
                self.delta = delta
                self.finish_reason = finish_reason
                self.index = index
        class MockStreamChunk: # Represents what litellm.acompletion might yield
            def __init__(self, choices, usage=None, id=None, model=None): # id and model for some providers
                self.choices = choices
                self.usage = usage
                self.id = id # Stream ID, not tool call ID
                self.model = model
        class MockUsage:
            def __init__(self, prompt_tokens=0, completion_tokens=0):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens

        original_request = MessagesRequest(
            model="anthropic/claude-3-opus-20240229",
            max_tokens=1024, messages=[], stream=True,
            tools=[Tool(name="tool_A", description="A tool", input_schema={"type":"object"})] # For tool_use case
        )
        
        # Mock the async generator that handle_streaming expects
        response_generator_mock = unittest.mock.AsyncMock() # This itself is not an async iterator yet

        # Define the sequence of chunks our mock generator will yield
        # This simulates a more complex interaction: text -> tool_call -> text -> finish
        chunks_to_yield = [
            MockStreamChunk(id="stream_id_1", choices=[MockChoice(index=0, delta=MockDelta(content="First text part. "))]),
            MockStreamChunk(id="stream_id_1", choices=[MockChoice(index=0, delta=MockDelta(tool_calls=[
                MockToolCallDelta(index=0, id="tool_call_123", function=MockFunctionDelta(name="tool_A", arguments='{"param":'))
            ]))]),
            MockStreamChunk(id="stream_id_1", choices=[MockChoice(index=0, delta=MockDelta(tool_calls=[
                MockToolCallDelta(index=0, id="tool_call_123", function=MockFunctionDelta(arguments='"value"'))
            ]))]),
            MockStreamChunk(id="stream_id_1", choices=[MockChoice(index=0, delta=MockDelta(tool_calls=[
                MockToolCallDelta(index=0, id="tool_call_123", function=MockFunctionDelta(arguments='}'))
            ]))]),
            # According to Anthropic spec, after tool_use, there should be a stop_reason "tool_use"
            # and then potentially more content if the model continues or another tool_call.
            # For this test, we'll assume the model finishes after one tool_use.
            MockStreamChunk(id="stream_id_1", choices=[MockChoice(index=0, delta=MockDelta(), finish_reason="tool_calls")], usage=MockUsage(prompt_tokens=50, completion_tokens=20)),
        ]

        # Make the mock an async iterator
        async def aiter_mock(): # Corrected typo here
            for chunk in chunks_to_yield:
                yield chunk
        response_generator_mock.__aiter__ = MagicMock(return_value=aiter_mock())


        results = []
        raw_sse_events = []
        async for item in conversion.handle_streaming(response_generator_mock, original_request):
            raw_sse_events.append(item)
            # Basic parsing for verification - real client would parse SSE properly
            if item.startswith("event:"):
                results.append(item.split("\n")[0])
            elif item.startswith("data:"):
                data_content = item.split("\n")[0][len("data: "):]
                if data_content == "[DONE]":
                    results.append("data: [DONE]")
                else:
                    try:
                        parsed_data = json.loads(data_content)
                        # Add type for easier assertion if it's a known event data structure
                        if 'type' in parsed_data:
                             results.append(f"data_type: {parsed_data['type']}")
                        if 'delta' in parsed_data and 'type' in parsed_data['delta']:
                             results.append(f"delta_type: {parsed_data['delta']['type']}")
                        if 'content_block' in parsed_data and 'type' in parsed_data['content_block']:
                             results.append(f"content_block_type: {parsed_data['content_block']['type']}")

                    except json.JSONDecodeError:
                        pass # Ignore non-json data lines for this simplified check
        
        # print("\nRAW SSE events for streaming test:\n" + "".join(raw_sse_events)) # For debugging

        # Assertions on the generated SSE events string list 'results'
        self.assertIn("event: message_start", results)
        self.assertIn("event: ping", results)
        
        # Text content part (might be empty if tool use is first)
        self.assertIn("event: content_block_start", results) # Initial text block
        self.assertIn("data_type: content_block_start", results)
        self.assertIn("content_block_type: text", results)
        self.assertIn("event: content_block_delta", results)
        self.assertIn("delta_type: text_delta", results) # For "First text part. "
        self.assertIn("event: content_block_stop", results) # Text block stops before tool_use

        # Tool call part
        self.assertIn("content_block_type: tool_use", results) # Start of tool_use
        self.assertIn("delta_type: input_json_delta", results) # For tool arguments
        
        # Message Delta and Stop
        self.assertIn("event: message_delta", results)
        self.assertIn("delta_type: message_delta", results) # Contains stop_reason and usage
        self.assertIn("event: message_stop", results)
        self.assertIn("data: [DONE]", results)

        # Check specific data payloads if necessary (more complex assertions)
        message_start_data = None
        content_block_starts = []
        content_block_deltas = []
        message_delta_data = None

        for event_str in raw_sse_events:
            if event_str.startswith("data:"):
                try:
                    data = json.loads(event_str[len("data: "):].strip())
                    if data.get("type") == "message_start": message_start_data = data
                    if data.get("type") == "content_block_start": content_block_starts.append(data)
                    if data.get("type") == "content_block_delta": content_block_deltas.append(data)
                    if data.get("type") == "message_delta": message_delta_data = data
                except: pass
        
        self.assertIsNotNone(message_start_data)
        self.assertEqual(message_start_data["message"]["model"], original_request.model)

        # Expecting two content_block_start: one for initial (possibly empty) text, one for tool_use
        self.assertTrue(len(content_block_starts) >= 1) # At least one for text
        # The refined logic in handle_streaming aims for: text_start -> text_delta -> text_stop -> tool_start -> tool_delta -> tool_stop
        tool_use_block_start = next((cb for cb in content_block_starts if cb["content_block"]["type"] == "tool_use"), None)
        self.assertIsNotNone(tool_use_block_start)
        if tool_use_block_start: # mypy check
             self.assertEqual(tool_use_block_start["content_block"]["name"], "tool_A")
             self.assertEqual(tool_use_block_start["index"], 1) # Assuming text block was index 0

        # Check some delta content
        text_deltas = [d["delta"]["text"] for d in content_block_deltas if d["delta"]["type"] == "text_delta"]
        self.assertIn("First text part. ", "".join(text_deltas))
        
        input_json_deltas = [d["delta"]["partial_json"] for d in content_block_deltas if d["delta"]["type"] == "input_json_delta"]
        self.assertIn('{"param":', "".join(input_json_deltas))
        self.assertIn('"value"}', "".join(input_json_deltas))

        self.assertIsNotNone(message_delta_data)
        self.assertEqual(message_delta_data["delta"]["stop_reason"], "tool_use")
        self.assertEqual(message_delta_data["usage"]["output_tokens"], mock_token_counter.return_value * 3) # 3 deltas with tokens counted by mock

    # Run the async part of the test
    asyncio.run(async_test_logic())

if __name__ == '__main__':
    unittest.main(verbosity=2)
