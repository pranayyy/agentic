import json
import re


def parse_llm_json(text: str, fallback: dict | None = None) -> dict:
    """
    Robustly parse JSON from an LLM response.
    Handles plain JSON, markdown code blocks, and partial JSON objects.
    """
    if fallback is None:
        fallback = {}

    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code block  ```json ... ```
    block_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Bare JSON object or array embedded in text
    obj_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(1))
        except json.JSONDecodeError:
            pass

    return fallback
