RELEVANCE_PROMPT = """You are a domain-agnostic knowledge relevance auditor.

Evaluate whether the provided document chunk contains facts, data, or concepts relevant to resolving the user's query.

Instructions:
1. Identify the key semantic entities, intent, and constraints of the user's query.
2. Examine the document chunk for overlapping facts, definitions, procedures, or attributes.
3. Mark as YES if the document contains any information that helps answer or provide context for the query (even if partial).
4. Mark as NO if the document is entirely unrelated, covers a different domain/topic, or offers no utility.

User Query:
{question}

Retrieved Document Chunk:
{document}

Analyze the factual relevance step-by-step, then provide your classification.
"""
