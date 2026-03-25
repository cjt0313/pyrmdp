import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def generate_background_knowledge(
    predicates_list: list,
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Call an LLM to generate Z3 axioms from a list of predicates.

    Parameters
    ----------
    predicates_list : list
        Predicate names from the PPDDL domain.
    llm_fn : callable, optional
        ``fn(prompt) → response_text``.  If *None*, builds one from
        ``llm.yaml`` / env-vars via :func:`synthesis.llm_config.build_llm_fn`.

    Returns
    -------
    str
        Axiom strings, one per line, e.g.
        ``FORALL x, y: IF on(x, y) THEN NOT clear(y)``
    """
    prompt = f"""Given the following relational MDP predicates: {predicates_list}
Generate logical axioms (background knowledge) to prune impossible states 
(e.g., Block A cannot be On Block B and Clear at the same time).
Output ONLY custom strings like:
FORALL x, y: IF on(x, y) THEN NOT clear(y)
"""

    if llm_fn is None:
        try:
            from pyrmdp.synthesis.llm_config import build_llm_fn
            llm_fn = build_llm_fn()
        except (EnvironmentError, ImportError) as exc:
            logger.warning(
                f"LLM unavailable ({exc}) — returning demo axioms."
            )
            return "FORALL x, y: IF on(x, y) THEN NOT clear(y)\n"

    return llm_fn(prompt)
