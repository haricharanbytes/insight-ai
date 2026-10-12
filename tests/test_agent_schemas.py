import inspect
import json

from src.agent.schemas import TOOL_SCHEMAS, openai_tools
from src.agent.tools import TOOLS
from src.analysis.filtering import FILTER_OPS


def test_every_tool_has_a_schema_and_vice_versa():
    assert set(TOOL_SCHEMAS) == set(TOOLS)


def test_schema_parameters_match_function_signatures():
    for name, fn in TOOLS.items():
        sig = inspect.signature(fn).parameters
        sig_params = {p for p in sig if p != "df"}
        props = set(TOOL_SCHEMAS[name]["parameters"]["properties"])
        required = set(TOOL_SCHEMAS[name]["parameters"]["required"])
        sig_required = {p for p in sig_params if sig[p].default is inspect.Parameter.empty}

        assert props == sig_params, f"{name}: schema and signature differ"
        assert required <= props, f"{name}: required names an unknown parameter"
        assert sig_required <= required, f"{name}: a mandatory parameter is not marked required"


def test_filter_operator_enum_is_not_hardcoded():
    items = TOOL_SCHEMAS["filter_data"]["parameters"]["properties"]["filters"]["items"]
    assert items["properties"]["op"]["enum"] == list(FILTER_OPS)


def test_openai_tools_format_and_serializable():
    tools = openai_tools()
    assert len(tools) == len(TOOLS)
    for t in tools:
        assert t["type"] == "function"
        assert t["function"]["name"] in TOOLS
        assert t["function"]["description"]
        assert t["function"]["parameters"]["type"] == "object"
    json.dumps(tools)