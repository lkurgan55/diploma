"""Prompt construction utilities."""

def build_prompt(tokenizer, question: str, schema_text: str | None = None, dialect: str = 'SQLite') -> str:
    """Builds a prompt for the model based on the question and schema."""
    system = (
        f"You are a Text-to-SQL generator for {dialect} dialect.\n"
        "Use only table and column names exactly as in the schema.\n"
        "Return EXACTLY ONE SQL query and NOTHING ELSE.\n"
        "Do NOT use markdown or code fences.\n"
        "Start the query with SELECT or WITH.\n"
        "End your output with a single semicolon ';' and then stop.\n"
        "No explanations, no comments."
    )
    if schema_text:
        user = (
            f"Schema:\n{schema_text}\n\n"
            f"Instruction: {question}\n"
            f"Output: SQL only, one statement, end with ';'"
        )
    else:
        user = (
            f"Instruction: {question}\n"
            f"Output: SQL only, one statement, end with ';'"
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
