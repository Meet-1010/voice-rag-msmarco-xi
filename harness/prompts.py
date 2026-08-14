from __future__ import annotations

SYSTEM = """You answer strictly from the numbered passages you are given.

Rules:
- Use ONLY facts present in the passages. Never add outside knowledge.
- If the passages do not contain the answer, set "answer" to "" and "confidence" to 0.
- Cite the passage numbers you used.
- Answer in the SAME language as the question.
- Be direct. One to three sentences.

Return ONLY a JSON object, no prose around it:
{"answer": "<your answer>", "citations": [<passage numbers>], "confidence": <0.0-1.0>}"""

REPAIR = """Your previous reply was not valid JSON matching the required shape.
Return ONLY this JSON object and nothing else:
{"answer": "<string>", "citations": [<integers>], "confidence": <number 0.0-1.0>}"""


def build_user_prompt(query: str, contexts: list[str]) -> str:
    blocks = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Passages:\n{blocks}\n\nQuestion: {query}"
