# Import the desired names from the neo4j module within this package
from .logger import (
    log_info,
    log_error,
    log_warning,
    log_debug
)

from .model import (
    initialize_models,
    get_spacy_model,
    get_fastcoref_model,
    get_gen_model,
    get_gen_tokenizer
)

from .processor import (
    generate_paragraph,
    extract_entities,
    extract_relations,
    resolve_coreferences,
    format_graph_to_text
)

# Optional: Define __all__ to control 'from app.db import *' behavior
# This explicitly lists what gets imported on wildcard imports. It's good practice.
__all__ = [
    log_info,
    log_error,
    log_warning,
    log_debug,
    initialize_models,
    get_spacy_model,
    get_fastcoref_model,
    get_gen_tokenizer,
    get_gen_model,
    generate_paragraph,
    extract_entities,
    extract_relations,
    resolve_coreferences,
    format_graph_to_text
]