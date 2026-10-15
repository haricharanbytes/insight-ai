import copy
import json

from src.agent.llm_client import LLMResponse


class ScriptedClient:
    """Stands in for LLMClient: returns scripted replies and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(message=reply, model="fake", usage={"prompt_tokens": 10, "completion_tokens": 5},
                           latency_s=0.01, attempts=1)


def tool_msg(name, args, cid="c1", raw_arguments=None):
    arguments = raw_arguments if raw_arguments is not None else json.dumps(args)
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def text_msg(text):
    return {"role": "assistant", "content": text}

def json_msg(obj):
    import json as _json
    return {"role": "assistant", "content": _json.dumps(obj)}