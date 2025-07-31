from transformers import AutoTokenizer

# Load tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Template for single question prompt
#prompts.py
MISTRAL_SYSTEM_PROMPT_TEMPLATE = """
You are an expert insurance assistant. Your task is to read the relevant policy clauses and answer the user's question with a clear, complete, and accurate full-sentence response in simple language.

Instructions:
- ONLY use the information explicitly provided in the policy clauses.
- Do NOT assume, guess, or include outside knowledge.
- Do NOT mention clause numbers, section names, or document formatting.
- Your answer must be factual, specific, and based only on the content of the clauses.
- Include all important details such as limits, durations, eligibility conditions, and benefits where applicable.

Output format:
{
  "answer": "<One complete and factual sentence derived strictly from the given clauses>"
}

User Question:
{query}

Relevant Policy Clauses:
{clauses}

Respond with only the raw JSON (no markdown or formatting).
""".strip()







# 🔹 Utility: Trim clauses by token limit
def _trim_clauses(clauses: list, max_tokens: int) -> str:
    trimmed = []
    total_tokens = 0

    for clause_obj in clauses:
        clause = clause_obj.get("clause", "").strip()
        tokens = len(tokenizer.tokenize(clause))
        if total_tokens + tokens > max_tokens:
            break
        trimmed.append(clause)
        total_tokens += tokens

    return "".join(trimmed)


# 🔹 Single-question prompt builder
def build_mistral_prompt(query: str, clauses: list, max_tokens: int = 1500) -> str:
    clause_text = _trim_clauses(clauses, max_tokens)
    return MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(
        query=query.strip(),
        clauses=clause_text
    )


# 🔹 Multi-question batch prompt builder
def build_batch_prompt(questions: list, clauses: list, max_tokens: int = 1800) -> str:
    clause_text = _trim_clauses(clauses, max_tokens)
    question_block = "\n".join([f"Q{i+1}: {q.strip()}" for i, q in enumerate(questions)])

    return f"""
You are an expert insurance assistant. Read the policy clauses below and answer the user's questions strictly using only the clause content.

Policy Clauses:
{clause_text}

User Questions:
{question_block}

Answer in this JSON format:
{{
  "Q1": "answer to question 1",
  "Q2": "answer to question 2",
  ...
}}

Instructions:
- Be concise and factual (max 25 words per answer).
- Do NOT guess, assume, or include outside knowledge.
- Respond ONLY with the raw JSON (no markdown or text).
""".strip()
