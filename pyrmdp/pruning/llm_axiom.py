import os

def generate_background_knowledge(predicates_list: list) -> str:
    """
    Call an LLM (OpenAI or Azure AI) to generate Z3 axioms from a list of predicates.
    The output should perfectly describe the constraints (e.g., blocks world logic).
    Returns a custom string format that can be parsed into Z3 expressions.
    """
    prompt = f"""
Given the following relational MDP predicates: {predicates_list}
Generate logical axioms (background knowledge) to prune impossible states (e.g., Block A cannot be On Block B and Clear at the same time).
Output ONLY custom strings like:
FORALL x, y: IF on(x, y) THEN NOT clear(y)
"""
    
    # Placeholder for actual LLM API call
    if "OPENAI_API_KEY" in os.environ:
        # call openai...
        return "FORALL x, y: IF on(x, y) THEN NOT clear(y)\n"
    elif "AZURE_OPENAI_API_KEY" in os.environ:
        # call azure...
        return "FORALL x, y: IF on(x, y) THEN NOT clear(y)\n"
    else:
        # Mocking output for demo purposes based on Blocks World
        return "FORALL x, y: IF on(x, y) THEN NOT clear(y)\n"
