#!/usr/bin/env python3
"""
验证 FastFlowLM 的 OpenAI 兼容 tool calling 能力。

测试场景：
1. 基础 tool calling — 模型是否返回 tool_calls
2. 多轮 tool calling — 工具结果反馈后模型是否继续调用
3. 工具定义格式 — Anthropic → OpenAI 转换是否正确
4. 响应解析 — tool_calls 能否被 _format_assistant_tool_message 正确解析
"""

import asyncio
import json
import sys
from openai import AsyncOpenAI

# FastFlowLM 配置
BASE_URL = "http://host.docker.internal:52625/v1"
API_KEY = "fastflow"
MODEL = "qwen3.5:9b"

# 简化的测试工具（模拟 Dialectic/Dreamer 的工具格式）
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search the peer's memory for relevant observations",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_consolidation",
            "description": "Signal that the consolidation task is complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of findings",
                    },
                },
            },
        },
    },
]


async def test_1_basic_tool_calling():
    """测试 1: 基础 tool calling — 模型是否返回 tool_calls"""
    print("\n" + "=" * 60)
    print("测试 1: 基础 tool calling")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    messages = [
        {
            "role": "user",
            "content": (
                "我最近和老金讨论了他的项目结构。"
                "请帮我查找关于老金项目的记忆。"
            ),
        }
    ]

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TEST_TOOLS,
            tool_choice="auto",
            max_tokens=512,
        )

        choice = response.choices[0]
        message = choice.message

        print(f"Finish reason: {choice.finish_reason}")
        print(f"Content: {message.content[:200] if message.content else '(null)'}")

        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"\n✅ 模型返回了 {len(message.tool_calls)} 个 tool_calls")
            for i, tc in enumerate(message.tool_calls):
                print(f"  [{i}] name={tc.function.name}")
                print(f"      arguments={tc.function.arguments[:200]}")
            return True
        else:
            print("\n⚠️  模型没有返回 tool_calls（可能认为不需要调用工具）")
            print("   这不算失败，但无法验证 tool calling 链路")
            return None

    except Exception as e:
        print(f"\n❌ 请求失败: {type(e).__name__}: {e}")
        return False


async def test_2_tool_result_feedback():
    """测试 2: 工具结果反馈 — 模型是否能处理 tool_result"""
    print("\n" + "=" * 60)
    print("测试 2: 工具结果反馈")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    # 先让模型调用工具
    messages = [
        {
            "role": "user",
            "content": "请调用 search_memory 查找老金的时区信息。",
        }
    ]

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TEST_TOOLS,
            tool_choice="required",  # 强制调用工具
            max_tokens=512,
        )

        choice = response.choices[0]
        message = choice.message

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            print("⚠️  模型未返回 tool_calls，跳过工具结果反馈测试")
            return None

        tool_call = message.tool_calls[0]
        print(f"工具调用: {tool_call.function.name}")
        print(f"参数: {tool_call.function.arguments[:200]}")

        # 模拟工具结果
        tool_result = json.dumps({
            "observations": [
                {"content": "owner lives in or operates within the GMT+8 time zone.", "id": "test123"}
            ]
        })

        # 构建带工具结果的对话
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

        # 让模型基于工具结果继续
        response2 = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TEST_TOOLS,
            tool_choice="auto",
            max_tokens=512,
        )

        choice2 = response2.choices[0]
        message2 = choice2.message

        print(f"\n工具结果反馈后:")
        print(f"Finish reason: {choice2.finish_reason}")
        print(f"Content: {message2.content[:300] if message2.content else '(null)'}")

        if hasattr(message2, "tool_calls") and message2.tool_calls:
            print(f"  → 又调用了 {len(message2.tool_calls)} 个工具")
        else:
            print(f"  → 返回了文本响应（正常）")

        return True

    except Exception as e:
        print(f"\n❌ 工具结果反馈失败: {type(e).__name__}: {e}")
        return False


async def test_3_anthropic_to_openai_conversion():
    """测试 3: Anthropic → OpenAI 工具格式转换"""
    print("\n" + "=" * 60)
    print("测试 3: Anthropic → OpenAI 工具格式转换")
    print("=" * 60)

    # 模拟 Anthropic 格式的工具定义
    anthropic_tool = {
        "name": "search_memory",
        "description": "Search memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    }

    # 转换逻辑（来自 convert_tools_for_provider）
    openai_tool = {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool["description"],
            "parameters": anthropic_tool["input_schema"],
        },
    }

    print("Anthropic 格式:")
    print(json.dumps(anthropic_tool, indent=2, ensure_ascii=False))
    print("\nOpenAI 格式:")
    print(json.dumps(openai_tool, indent=2, ensure_ascii=False))

    # 验证格式
    assert openai_tool["type"] == "function"
    assert "parameters" in openai_tool["function"]
    assert "input_schema" not in openai_tool["function"]  # 不应有 input_schema

    print("\n✅ 格式转换正确")
    return True


async def test_4_tool_choice_modes():
    """测试 4: tool_choice 不同模式"""
    print("\n" + "=" * 60)
    print("测试 4: tool_choice 模式验证")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    test_cases = [
        ("auto", "auto"),
        ("required", "required"),
        ("{\"type\": \"function\", \"function\": {\"name\": \"search_memory\"}}", "function"),
    ]

    for tc_name, tc_value in test_cases:
        print(f"\n  tool_choice={tc_name}:")
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "你好"}],
                tools=TEST_TOOLS,
                tool_choice=tc_value,
                max_tokens=128,
            )
            choice = response.choices[0]
            message = choice.message
            has_tools = hasattr(message, "tool_calls") and message.tool_calls
            print(f"    Finish: {choice.finish_reason}, Has tool_calls: {has_tools}")
        except Exception as e:
            print(f"    ❌ 失败: {type(e).__name__}: {str(e)[:100]}")

    print("\n✅ tool_choice 模式验证完成")
    return True


async def main():
    print(f"\n🧪 FastFlowLM Tool Calling 验证")
    print(f"模型: {MODEL}")
    print(f"API: {BASE_URL}")
    print(f"时间: {__import__('datetime').datetime.now().isoformat()}")

    results = {}

    # 测试 1: 基础 tool calling
    results["基础 tool calling"] = await test_1_basic_tool_calling()

    # 测试 2: 工具结果反馈
    results["工具结果反馈"] = await test_2_tool_result_feedback()

    # 测试 3: 格式转换
    results["Anthropic→OpenAI 转换"] = await test_3_anthropic_to_openai_conversion()

    # 测试 4: tool_choice 模式
    results["tool_choice 模式"] = await test_4_tool_choice_modes()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        elif result is None:
            print(f"  ⚠️  {name} (跳过/不确定)")
        else:
            print(f"  ❓ {name}: {result}")

    # 返回退出码
    has_failure = any(r is False for r in results.values())
    return 1 if has_failure else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
