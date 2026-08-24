from openai import OpenAI

from app.config import settings

_client = OpenAI(base_url=settings.vllm_base_url, api_key="not-needed")

SYSTEM_PROMPT = (
    "You are a Dungeons & Dragons rules assistant. Answer the user's question using ONLY "
    "the excerpts provided below. If the excerpts don't contain the answer, say so plainly "
    "instead of guessing. After your answer, cite every excerpt you used in the form "
    "[filename, p.X, section]. Respond in the same language the user asked their question in, "
    "even though the excerpts themselves are in English."
)


def _format_pages(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return "p.?"
    if page_end is None or page_start == page_end:
        return f"p.{page_start}"
    return f"pp.{page_start}-{page_end}"


def _build_prompt(question: str, chunks: list[dict]) -> str:
    excerpts = []
    for c in chunks:
        pages = _format_pages(c.get("page_start"), c.get("page_end"))
        heading = c.get("section_heading") or ""
        excerpts.append(f"[{c['filename']}, {pages}, {heading}]\n{c['content']}")
    joined = "\n\n---\n\n".join(excerpts)
    return f"Excerpts:\n\n{joined}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> str:
    user_prompt = _build_prompt(question, chunks)
    response = _client.chat.completions.create(
        model=settings.vllm_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
