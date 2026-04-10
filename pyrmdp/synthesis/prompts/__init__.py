"""
pyrmdp.synthesis.prompts — Prompt templates for LLM-based tasks.

Each file in this package handles ONE LLM query purpose:

  • ``vlm_domain_prompt``    — Step 0a: RGB → types + predicates
  • ``llm_operator_prompt``  — Step 0b: task NL → operators
  • ``llm_failure_prompt``   — Step 1:  failure hallucination
  • ``llm_recovery_prompt``  — Step 5:  recovery operator synthesis
  • ``llm_mutex_prompt``     — R5:      mutex constraint generation
  • ``response_parser``      — Shared JSON extraction utility

Every prompt module exposes:
  ``build_*_prompt(…)  → {"system": str, "user": str}``
  ``parse_*_response(…) → dict | list | None``
"""
