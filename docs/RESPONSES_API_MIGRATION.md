# Migrating to the OpenAI Responses API

The OpenAI Responses API replaces the older Assistants and Threads APIs. MACS
already uses the Responses API through `llm_openai.py`, but older environment
variables and mental models can still cause confusion. This guide explains what
changed and how to update your setup.

## What Changed

- Assistants API used `assistant_id` and `thread_id`.
- Responses API uses `prompt_id` (reusable instructions) and `conversation_id`
  (ongoing session state).
- You can store these IDs in environment variables or inside `config.json`.

## Step-by-Step Update for MACS

1. Keep your OpenAI API key in one of these places:
   - Environment variable: `OPENAI_API_KEY`
   - `config.json` under `system_variables.openai_api_key`

2. Rename environment variables if you still use the old ones:
   - `OPENAI_ASSISTANT_ID` -> `OPENAI_PROMPT_ID`
   - `OPENAI_THREAD_ID` -> `OPENAI_CONVERSATION_ID`

3. If you prefer to keep prompt IDs in config, add them here:

```json
{
  "system_variables": {
    "openai_api_key": "sk-...",
    "prompt_ids": ["pr_123"]
  }
}
```

4. Run a quick validation:

```bash
python -c "from llm_openai import OpenAILLM; OpenAILLM()"
```

If the client initializes without errors, your setup is correct.

## Example Responses API Call

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

## Common Mistakes and Fixes

- Mistake: Keeping `OPENAI_ASSISTANT_ID` and `OPENAI_THREAD_ID`.
  Fix: Rename them to `OPENAI_PROMPT_ID` and `OPENAI_CONVERSATION_ID`.
- Mistake: Using an old OpenAI library.
  Fix: Run `python -m pip install -r requirements.txt`.
- Mistake: Leaving `openai_api_key` empty in `config.json`.
  Fix: Add a valid key or set `OPENAI_API_KEY` in your shell.
