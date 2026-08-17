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


# Used only when retrieval found nothing usable and the caller allowed a
# fallback. The corpus is not mentioned because there is no corpus content to
# ground against - pretending otherwise is how a system starts inventing
# citations.
GENERAL_SYSTEM = """You are a helpful assistant answering from your own general knowledge.

Rules:
- Answer directly and factually. Two to four sentences.
- Answer in the SAME language as the question.
- If you are genuinely unsure, say so rather than inventing specifics.
- Do not cite sources; you have none for this answer.

Return ONLY a JSON object:
{"answer": "<your answer>", "citations": [], "confidence": <0.0-1.0>}"""


def build_general_prompt(query: str) -> str:
    return f"Question: {query}"


def build_user_prompt(query: str, contexts: list[str]) -> str:
    blocks = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Passages:\n{blocks}\n\nQuestion: {query}"
