# Migrating to the OpenAI Responses API

The OpenAI Responses API replaces the earlier Assistants and Threads APIs. Prompts and conversations now identify
reusable instructions and ongoing sessions.

## Environment Variables

If you previously used the Assistants API, rename any environment variables as follows:

- `OPENAI_ASSISTANT_ID` → `OPENAI_PROMPT_ID`
- `OPENAI_THREAD_ID` → `OPENAI_CONVERSATION_ID`

`OPENAI_PROMPT_ID` and `OPENAI_CONVERSATION_ID` may be supplied directly as environment variables or stored in
`system_variables.prompt_ids` inside `config.json`.

Example configuration snippet:

```json
{
  "system_variables": {
    "openai_api_key": "sk-...",
    "prompt_ids": ["pr_123"]
  }
}
```

## Example Usage

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"},
    ],
    prompt_id="pr_123",
    conversation_id="cv_456",
)
print(response.output_text)
```

This call returns a response object whose `output_text` contains the model's reply.
