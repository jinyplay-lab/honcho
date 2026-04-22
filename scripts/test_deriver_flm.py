#!/usr/bin/env python3
"""
Simulate deriver call to FLM to debug parsing logic.

Usage:
    python3 scripts/test_deriver_flm.py
"""

import asyncio
import json
import sys

sys.path.insert(0, "/home/node/.openclaw/workspace/containers/honcho/honcho")

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional

# FLM configuration
FLM_BASE_URL = "http://host.docker.internal:52625/v1"
FLM_API_KEY = "fastflow"
MODEL = "qwen3.5:9b"


class ExplicitObservation(BaseModel):
    observation: str = Field(description="A single explicit fact about the user")


class PromptRepresentation(BaseModel):
    explicit: list[ExplicitObservation] = Field(
        default_factory=list,
        description="List of explicit observations about the user",
    )
    deductive: Optional[list] = Field(
        default=None,
        description="List of deductive observations about the user",
    )


async def test_flm_call():
    client = AsyncOpenAI(
        api_key=FLM_API_KEY,
        base_url=FLM_BASE_URL,
    )

    messages = """
User: Hi, I just moved to New York City last week.
User: I'm 28 years old and work as a software engineer.
User: My dog's name is Max and he's a golden retriever.
"""

    prompt = f"""
Analyze messages from user to extract **explicit atomic facts** about them.

[EXPLICIT] DEFINITION: Facts about user that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about user, say so. If user is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand user.
- Extract ALL observations from user messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

Messages to analyze:
<messages>
{messages}
</messages>
"""

    print("=" * 60)
    print("TEST 1: Standard create (no response_format)")
    print("=" * 60)
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        content = resp.choices[0].message.content or ""
        print(f"finish_reason: {resp.choices[0].finish_reason}")
        print(f"content length: {len(content)}")
        print(f"content preview: {content[:500]}")
        print()

        # Try to parse as JSON
        print("TEST 2: JSON parse attempt")
        print("-" * 40)
        try:
            # Try to extract JSON from content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)
            print(f"✅ JSON parsed successfully!")
            print(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse failed: {e}")
            print(f"Content starts with: {content[:100]}")
        print()

    except Exception as e:
        print(f"❌ Request failed: {e}")
        print()

    # Test with response_format
    print("TEST 3: With response_format (strict JSON)")
    print("=" * 60)
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        print(f"finish_reason: {resp.choices[0].finish_reason}")
        print(f"content length: {len(content)}")
        print(f"content: {content[:500]}")
        print()

        # Try to parse as JSON
        print("TEST 4: JSON parse with response_format")
        print("-" * 40)
        try:
            data = json.loads(content.strip())
            print(f"✅ JSON parsed successfully!")
            print(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse failed: {e}")
            print(f"Content starts with: {content[:100]}")
        print()

    except Exception as e:
        print(f"❌ Request failed: {e}")
        print()

    # Test with Pydantic model parsing
    print("TEST 5: Pydantic model validation")
    print("=" * 60)
    try:
        # Use the content from test 1 or 3
        test_content = content if 'content' in dir() else ""
        if not test_content:
            print("No content to test")
            return

        # Try to extract JSON
        if "```json" in test_content:
            json_str = test_content.split("```json")[1].split("```")[0].strip()
        elif "```" in test_content:
            json_str = test_content.split("```")[1].split("```")[0].strip()
        else:
            json_str = test_content.strip()

        data = json.loads(json_str)
        representation = PromptRepresentation(**data)
        print(f"✅ Pydantic validation succeeded!")
        print(f"Explicit observations: {len(representation.explicit)}")
        for obs in representation.explicit:
            print(f"  - {obs.observation}")
    except Exception as e:
        print(f"❌ Pydantic validation failed: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(test_flm_call())
