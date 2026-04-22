#!/usr/bin/env python3
"""
验证 prompt-based 工具调用模拟 (FastFlowLM fallback)。

测试场景：
1. _parse_text_for_tool_calls 解析正确性（内联实现，不依赖完整模块）
2. 模型实际输出格式
3. 多轮工具调用模拟
"""

import asyncio
import json
import re
import sys
import uuid
from openai import AsyncOpenAI

BASE_URL = "http://host.docker.internal:52625/v1"
API_KEY = "fastflow"
MODEL = "qwen3.5:9b"

TEST_TOOLS = [
    {"name": "search_memory", "description": "Search memory", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "search_messages", "description": "Search messages", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "finish_consolidation", "description": "Finish task", "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}},
]


def _parse_tool_args(args_str):
    """Parse tool arguments: JSON or keyword args format."""
    try:
        args = json.loads(args_str)
        if isinstance(args, dict):
            return args
    except json.JSONDecodeError:
        pass

    # Try keyword args: key="value", key2='value', key3=unquoted
    kw_pattern = re.compile(
        r"""(?:(\w+)\s*=\s*)?(?:(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|([\S]+)))(?:\s*,\s*|$)""",
    )
    matches = kw_pattern.findall(args_str)
    if matches:
        args = {}
        for key_or_empty, dq, sq, uq in matches:
            key = key_or_empty if key_or_empty else "_arg"
            value = dq or sq or uq
            args[key] = value
        if args:
            return args

    return {"_raw": args_str}


def _parse_text_for_tool_calls(text, available_tools):
    """Parse tool calls from model text output (FastFlowLM fallback)."""
    tool_names = {t["name"] for t in available_tools}
    tool_calls = []
    remaining_lines = []
    tool_pattern = re.compile(r"^\s*\[TOOL:(\w+)\((.+)\)\]\s*$")

    for line in text.split("\n"):
        m = tool_pattern.match(line)
        if m and m.group(1) in tool_names:
            name = m.group(1)
            args_str = m.group(2)
            args = _parse_tool_args(args_str)
            tool_calls.append({
                "id": f"tc_{uuid.uuid4().hex[:12]}",
                "name": name,
                "input": args,
            })
        else:
            remaining_lines.append(line)

    remaining_text = "\n".join(remaining_lines).strip()
    return tool_calls, remaining_text


def test_parser():
    """测试解析器"""
    print("\n" + "=" * 60)
    print("测试 1: 解析器单元测试")
    print("=" * 60)

    tests = []

    # A: 单个 tool call
    text = '[TOOL:search_memory({"query": "老金的项目"})]'
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 1 and calls[0]["name"] == "search_memory"
    print(f"\n  A. 单个 tool call: {'✅' if ok else '❌'}")
    if ok:
        print(f"     → {calls[0]['name']}({json.dumps(calls[0]['input'])})")
    tests.append(("单个 tool call", ok))

    # B: 多个 tool calls (每行一个)
    text = '[TOOL:search_memory({"query": "老金"})]\n[TOOL:search_messages({"query": "项目"})]'
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 2
    print(f"\n  B. 多个 tool calls: {'✅' if ok else '❌'}")
    if ok:
        for c in calls:
            print(f"     → {c['name']}({json.dumps(c['input'])})")
    tests.append(("多个 tool calls", ok))

    # C: tool calls + 最终回答
    text = '[TOOL:search_memory({"query": "老金"})]\n\n根据搜索结果，老金是一个开发者。'
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 1 and remaining.strip().startswith("根据")
    print(f"\n  C. 混合输出: {'✅' if ok else '❌'}")
    print(f"     Tool calls: {len(calls)}, 剩余: '{remaining[:60]}'")
    tests.append(("混合输出", ok))

    # G: 关键字参数格式 (模型实际输出)
    text = '[TOOL:search_memory(query="老金 时区", limit=5)]'
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 1 and "query" in calls[0]["input"]
    print(f"\n  G. 关键字参数格式: {'✅' if ok else '❌'}")
    if ok:
        print(f"     → {calls[0]['name']}({json.dumps(calls[0]['input'])})")
    tests.append(("关键字参数", ok))

    # D: 纯文本（无 tool calls）
    text = "根据我的分析，老金喜欢用 Python。"
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 0 and remaining == text
    print(f"\n  D. 纯文本: {'✅' if ok else '❌'}")
    tests.append(("纯文本", ok))

    # E: 无效 tool name
    text = '[TOOL:unknown_tool({"query": "test"})]'
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 0 and remaining == text
    print(f"\n  E. 无效 tool name: {'✅' if ok else '❌'}")
    tests.append(("无效 tool name", ok))

    # F: 带缩进的 tool call
    text = "  [TOOL:finish_consolidation({\"summary\": \"done\"})]  "
    calls, remaining = _parse_text_for_tool_calls(text, TEST_TOOLS)
    ok = len(calls) == 1 and calls[0]["name"] == "finish_consolidation"
    print(f"\n  F. 带缩进: {'✅' if ok else '❌'}")
    if ok:
        print(f"     → {calls[0]['name']}({json.dumps(calls[0]['input'])})")
    tests.append(("带缩进", ok))

    all_passed = all(r[1] for r in tests)
    print(f"\n  {'✅ 全部通过' if all_passed else '❌ 有失败'}")
    return all_passed


async def test_model_output():
    """测试模型实际输出格式"""
    print("\n" + "=" * 60)
    print("测试 2: 模型实际输出格式")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    prompt = """你是 Dialectic 分析代理。你需要回答关于用户的问题。

可用工具：
- search_memory(query, limit=5): 搜索用户记忆
- search_messages(query): 搜索消息
- finish_consolidation(summary): 标记任务完成

当需要调用工具时，使用以下格式（每行一个）：
  [TOOL:tool_name({"arg": "value"})]

当收集完所有信息后，直接给出最终回答。
不要在同一个响应中混合 tool calls 和最终回答。

问题: 老金的时区是什么？请先调用 search_memory 查找相关信息。"""

    print(f"\n  发送请求到 {MODEL}...")
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        print(f"  Finish reason: {finish_reason}")
        print(f"\n  模型输出:")
        print(f"  {'─' * 50}")
        print(f"  {content.strip()}")
        print(f"  {'─' * 50}")

        calls, remaining = _parse_text_for_tool_calls(content, TEST_TOOLS)
        print(f"\n  解析结果:")
        print(f"    Tool calls: {len(calls)}")
        for c in calls:
            print(f"      - {c['name']}({json.dumps(c['input'])})")
        print(f"    剩余文本: '{remaining[:100]}'")

        return len(calls) > 0, content

    except Exception as e:
        print(f"  ❌ 请求失败: {type(e).__name__}: {e}")
        return False, str(e)


async def test_multi_turn():
    """模拟多轮工具调用"""
    print("\n" + "=" * 60)
    print("测试 3: 多轮工具调用模拟")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    prompt = """你是分析代理。使用工具收集信息。

可用工具：
- search_memory(query, limit=5): 搜索记忆
- finish_consolidation(summary): 完成任务

格式：[TOOL:name({"arg": "value"})]

请搜索关于"老金"的记忆。"""

    messages = [{"role": "user", "content": prompt}]
    max_iterations = 3

    for i in range(max_iterations):
        print(f"\n  轮次 {i + 1}/{max_iterations}:")
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )

            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})

            calls, remaining = _parse_text_for_tool_calls(content, TEST_TOOLS)

            if calls:
                print(f"    → 解析到 {len(calls)} 个 tool call(s)")
                for c in calls:
                    print(f"      [{c['name']}] {json.dumps(c['input'])}")
                    tool_result = f"Tool '{c['name']}' executed. Result: found 3 relevant observations."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": tool_result,
                    })
            else:
                print(f"    → 最终回答: {remaining[:150]}")
                print(f"    ✅ 多轮模拟完成")
                return True

        except Exception as e:
            print(f"    ❌ 失败: {type(e).__name__}: {e}")
            return False

    print(f"    ⚠️  达到最大轮次")
    return False


async def main():
    print(f"\n🧪 Prompt-based Tool Calling 验证")
    print(f"模型: {MODEL}")
    print(f"API: {BASE_URL}")
    print(f"时间: {__import__('datetime').datetime.now().isoformat()}")

    results = {}
    results["解析器单元测试"] = test_parser()
    has_tool_calls, raw = await test_model_output()
    results["模型输出含 tool calls"] = has_tool_calls
    results["多轮工具调用模拟"] = await test_multi_turn()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⚠️  {name}")

    return 1 if any(r is False for r in results.values()) else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
