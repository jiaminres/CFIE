# 2026-05-26 GUI Agent Inline Media Context Fix

## Problem

Live GUI Agent runs showed an abnormal `input_tokens=74027` for a single
1600x900 screenshot turn. That is not the expected visual-token cost. The same
screenshot was being sent twice:

- correctly as a typed `input_image`;
- incorrectly as a `data:image/...;base64,...` string embedded inside runtime
  context JSON text.

The second path makes the tokenizer treat the image bytes as ordinary text,
which destroys prefill latency and prefix-cache efficiency.

## Fix

- `cfie_client.responses_adapter`
  - Sanitizes text-bearing Responses input items before HTTP submission.
  - Keeps real typed media parts such as `input_image.image_url`.
- `cfie_gui_agent.openai_responses`
  - Uses the shared client adapter when normalizing GUI Agent conversations.
- `cfie.entrypoints.openai.responses.input_sanitizer`
  - Adds the same guard at the server Responses protocol boundary.
- `cfie.entrypoints.openai.responses.protocol`
  - Runs sanitizer during `ResponsesRequest` validation.
- `cfie.entrypoints.openai.responses.utils`
  - Sanitizes string input, instructions, and function-call outputs during
    conversion to chat messages.

## Rule

Strip only inline media blobs from text-bearing fields:

```text
data:image/...;base64,...
data:video/...;base64,...
```

Do not strip ordinary user text. Do not strip typed media parts:

```json
{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
```

## Verification

- `tests/unit/test_gui_agent_openai_responses.py`
- `tests/unit/test_responses_video_input.py`
- `tests/unit/test_openai_reasoning_template.py`
- `tests/unit/test_responses_tool_call_normalizer.py`
- `tests/unit/test_cfie_client_gui_agent.py`
- `tests/unit/test_gui_agent_architecture.py`
- `tests/unit/test_gui_agent_workflow.py`

Results:

```text
21 passed
83 passed
10 passed
```

Manual sanity check:

```text
developer_text_chars=78
developer_has_data_image=False
image_part_retained=True
```

This confirms that a large inline screenshot in runtime text is replaced with a
short placeholder while the real `input_image` remains available to the model.
