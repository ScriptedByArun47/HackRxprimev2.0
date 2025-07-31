from transformers import AutoTokenizer
from difflib import SequenceMatcher

# Load tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Template for single question prompt
MISTRAL_SYSTEM_PROMPT_TEMPLATE = """
You are an insurance assistant. Answer the user's question using only the policy clauses below.

Instructions:
- Be accurate and clear.
- Use only the clause content — no guessing or outside info.
- Write a short, simple sentence.
- Don’t mention formatting or section numbers.
- If the answer isn't found, say: "The policy doesn’t mention this."

Respond only in this JSON format:
{
  "answer": "<short, clear sentence based only on the clauses>"
}

Question:
{query}

Clauses:
{clauses}
""".strip()

# 🔹 Utility: Deduplicate clauses (semantic similarity)
def _deduplicate_clauses(clauses: list, threshold: float = 0.88) -> list:
    unique = []
    for clause_obj in clauses:
        clause = clause_obj.get("clause", "").strip()
        if not clause:
            continue
        if all(SequenceMatcher(None, clause, u.get("clause", "")).ratio() < threshold for u in unique):
            unique.append({"clause": clause})
    return unique


# 🔹 Utility: Trim clauses by token limit
def _trim_clauses(clauses: list, max_tokens: int) -> str:
    deduped = _deduplicate_clauses(clauses)
    trimmed = []
    total_tokens = 0

    for clause_obj in deduped:
        clause = clause_obj.get("clause", "").strip()
        tokens = len(tokenizer.tokenize(clause))
        if total_tokens + tokens > max_tokens:
            break
        trimmed.append(clause)
        total_tokens += tokens

    return "\n\n".join(trimmed)  # Preserve clause boundaries for LLM grounding


# 🔹 Single-question prompt builder
def build_mistral_prompt(query: str, clauses: list, max_tokens: int = 1000) -> str:
    clause_text = _trim_clauses(clauses, max_tokens)
    return MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(
        query=query.strip(),
        clauses=clause_text
    )


# 🔹 Multi-question batch prompt builder
def build_prompt_batch(question_clause_map):
    prompt_lines = []
    for i, (question, clauses) in enumerate(question_clause_map.items(), start=1):
        joined = " ".join(c["clause"].replace('"', '\\"') for c in clauses)
        prompt_lines.append(f'"Q{i}": {{"question": "{question}", "clauses": "{joined}"}}')

    json_data = "{\n" + ",\n".join(prompt_lines) + "\n}"

    prompt = (
        "You are an expert health insurance assistant. Answer each question strictly based ONLY on the provided clauses. "
        "If the answer cannot be found in the clauses, reply with 'The provided text does not contain this information.'\n\n"
        "Return your response as JSON with keys like 'Q1', 'Q2', each having an 'answer' field with a full sentence answer.\n"
        "Example format:\n"
        '{ "Q1": {"answer": "..."}, "Q2": {"answer": "..."} }\n\n'
        f"Entries:\n{json_data}"
    )
    return prompt


# 🧪 Optional: Token counter (for profiling)
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))
