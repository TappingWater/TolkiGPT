from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import torch
import app.config as config
from torch.cuda.amp import autocast
import app.utils as utils
from app.utils.logger import log_debug, log_error, log_info, log_warning
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from app.utils.model import get_fastcoref_model, get_spacy_model
from spacy.tokens import Span
import nltk

def generate_paragraph(text):
    prompt = text
    tokenizer = utils.get_gen_tokenizer()
    model = utils.get_gen_model()
    device = config.DEFAULT_INFERENCE_DEVICE
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        with autocast(device):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    next_paragraph = output_text.replace(prompt.strip(), "").strip()
    # Make sure the output is a full sentence ending with proper punctuation
    if not next_paragraph.endswith(('.', '!', '?')):
        next_paragraph = next_paragraph.rsplit('.', 1)[0] + '.'
    return next_paragraph

def extract_entities(text: str) -> Dict[str, str]:
    """Extract named entities with spaCy"""
    nlp = get_spacy_model()
    if nlp is None:
        log_error("spaCy model not initialized. Cannot extract entities.")
        return {}
    try:
        doc = nlp(text)
        # Return map of entity text to entity label
        return {ent.text.strip(): ent.label_ for ent in doc.ents}
    except Exception as e:
        log_error(f"Error during spaCy entity extraction: {e}", exc_info=True)
        return {}

# --- Existing Coreference Resolution ---
def resolve_coreferences(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Resolves coreferences using FastCoref and correlates with spaCy NER.
    Uses spaCy for initial tokenization to align indices.
    Improved logic for linking clusters to NER entities, checking exact and root matches.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary where keys are canonical
        named entity strings, and values are lists of mention dictionaries.
        Each mention dictionary contains 'text', 'start_char', 'end_char'.
        Example: {'Cinderella': [{'text': 'She', 'start_char': 10, 'end_char': 13}, ...]}
    """
    # 1. Get Models
    nlp = get_spacy_model()
    fc_model = get_fastcoref_model()

    if nlp is None or fc_model is None:
        log_error("spaCy or FastCoref model not initialized. Cannot resolve coreferences.")
        return {}

    if not text.strip():
        log_debug("Received empty text for structured coreference resolution.")
        return {}

    log_info(f"Starting structured coreference resolution for text: '{text[:100]}...'")

    try:
        # 2. Run spaCy NER and Tokenization
        doc = nlp(text)
        ner_entities_by_start_char: Dict[int, Span] = {ent.start_char: ent for ent in doc.ents}
        # Store canonical NER texts (stripped) in a set for quick lookups
        canonical_ner_texts = {ent.text.strip() for ent in doc.ents}
        log_debug(f"Found {len(canonical_ner_texts)} canonical NER entities: {canonical_ner_texts}")

        # --- Create list of tokens for FastCoref ---
        spacy_tokens = [token.text for token in doc]
        log_debug(f"spaCy tokenized text into {len(spacy_tokens)} tokens.")
        if not spacy_tokens:
             log_warning("spaCy tokenization resulted in zero tokens.")
             return {}

        # 3. Run FastCoref with pre-tokenized input
        log_debug("Running FastCoref prediction with spaCy tokens...")
        preds = fc_model.predict(texts=[spacy_tokens], is_split_into_words=True)
        log_debug("FastCoref prediction completed.")

        if not preds or not preds[0]:
            log_info("FastCoref returned no predictions.")
            return {}

        clusters_token_indices = preds[0].get_clusters(as_strings=False)
        log_info(f"FastCoref found {len(clusters_token_indices)} clusters (token indices).")

        # 4. Process Clusters and Map to NER Entities
        resolved_coreferences: Dict[str, List[Dict[str, Any]]] = {}

        for cluster in clusters_token_indices:
            if not cluster: continue

            cluster_mentions_data = [] # Store {'text': ..., 'start_char': ..., 'end_char': ...} for this cluster
            canonical_entity_for_cluster: Optional[str] = None
            # Track the best match found so far: ('match_type', length, text)
            # 'match_type': 0 for exact NER match, 1 for root token match
            best_match_info: Tuple[int, int, Optional[str]] = (2, -1, None) # (higher type = worse match, length, text)

            # --- First Pass: Convert all mentions and find the best NER anchor ---
            for mention_token_indices in cluster:
                start_token_idx, end_token_idx = mention_token_indices

                if start_token_idx < 0 or end_token_idx >= len(doc):
                    log_warning(f"Cluster mention token indices {mention_token_indices} out of bounds for doc length {len(doc)}. Skipping mention.")
                    continue

                try:
                    mention_span = doc[start_token_idx : end_token_idx + 1]
                except IndexError:
                     log_warning(f"IndexError accessing doc[{start_token_idx}:{end_token_idx + 1}] for doc length {len(doc)}. Skipping mention.")
                     continue

                mention_text = mention_span.text.strip()
                mention_start_char = mention_span.start_char
                mention_end_char = mention_span.end_char

                if not mention_text:
                    log_debug(f"Empty mention text for token indices {mention_token_indices}. Skipping.")
                    continue

                mention_data = {
                    "text": mention_text,
                    "start_char": mention_start_char,
                    "end_char": mention_end_char
                }
                cluster_mentions_data.append(mention_data)

                # --- Check for potential anchor match ---
                current_match_type = 2 # Default to no match
                current_match_text = None
                current_match_len = -1

                # 1. Check for exact NER span match
                if mention_start_char in ner_entities_by_start_char:
                     ner_span = ner_entities_by_start_char[mention_start_char]
                     if ner_span.end_char == mention_end_char and ner_span.text.strip() == mention_text:
                          # Exact match found
                          current_match_type = 0
                          current_match_text = ner_span.text.strip()
                          current_match_len = len(current_match_text)
                          log_debug(f"Mention '{mention_text}' is an exact NER match for '{current_match_text}'")

                # 2. If no exact match, check if root token text matches a canonical NER entity text
                if current_match_type > 0: # Only check if not already an exact match
                    root_token = mention_span.root
                    root_text = root_token.text.strip()
                    # Check against the set of known NER entity strings
                    if root_text in canonical_ner_texts:
                        # Root token matches an NER entity name
                        current_match_type = 1
                        current_match_text = root_text
                        current_match_len = len(current_match_text) # Use length of root token text
                        log_debug(f"Mention '{mention_text}' has root '{root_text}' matching NER entity '{current_match_text}'")

                # --- Update best match for the cluster ---
                if current_match_text is not None:
                    # Compare with best_match_info: prioritize lower type, then longer length
                    if current_match_type < best_match_info[0] or \
                       (current_match_type == best_match_info[0] and current_match_len > best_match_info[1]):
                        best_match_info = (current_match_type, current_match_len, current_match_text)
                        log_debug(f"Updating best anchor for cluster to '{current_match_text}' (type: {current_match_type}, len: {current_match_len})")

            # --- End of first pass for cluster ---
            canonical_entity_for_cluster = best_match_info[2] # Get the text of the best match

            # --- Second Pass: Add mentions to the map if a canonical entity was found ---
            if canonical_entity_for_cluster:
                # Ensure the list exists for this canonical entity
                if canonical_entity_for_cluster not in resolved_coreferences:
                    resolved_coreferences[canonical_entity_for_cluster] = []

                # Add all collected mention data for this cluster
                resolved_coreferences[canonical_entity_for_cluster].extend(cluster_mentions_data)
                log_debug(f"Linked cluster to canonical entity '{canonical_entity_for_cluster}'. Added {len(cluster_mentions_data)} mentions.")
            else:
                 # Only log if the cluster wasn't empty after potential skips
                 if cluster_mentions_data:
                      log_debug(f"Could not link cluster with mentions {[m['text'] for m in cluster_mentions_data]} to a canonical NER entity. Discarding cluster.")


        # Optional: Deduplicate mentions within each list if needed
        for entity, mentions in resolved_coreferences.items():
             unique_mentions = []
             seen_spans = set()
             for mention in mentions:
                  span_tuple = (mention['start_char'], mention['end_char'])
                  if span_tuple not in seen_spans:
                       unique_mentions.append(mention)
                       seen_spans.add(span_tuple)
             resolved_coreferences[entity] = unique_mentions
             if len(unique_mentions) < len(mentions):
                 log_debug(f"Deduplicated mentions for entity '{entity}'.")


        log_info(f"Structured coreference resolution successful. Found mappings for {len(resolved_coreferences)} entities.")
        return resolved_coreferences

    except Exception as e:
        log_error(f"Error during structured coreference resolution: {e}", exc_info=True)
        return {}

# --- NEW Helper: Resolve Token Span to Entity ---
def find_entity_for_span(
    span_start_char: int,
    span_end_char: int,
    resolved_coreferences: Dict[str, List[Dict[str, Any]]]
) -> Optional[str]:
    """
    Checks if a given character span matches any mention in the structured coreference map.

    Args:
        span_start_char: The starting character index of the text span.
        span_end_char: The ending character index of the text span.
        resolved_coreferences: The structured coreference map.
            Example: {'Cinderella': [{'text': 'She', 'start_char': 10, 'end_char': 13}, ...]}

    Returns:
        The canonical entity name if the span matches a mention, otherwise None.
    """
    # Iterate through each canonical entity in the map
    for canonical_entity, mentions in resolved_coreferences.items():
        # Iterate through each mention associated with this entity
        for mention in mentions:
            # Check if the start and end characters match exactly
            if mention['start_char'] == span_start_char and mention['end_char'] == span_end_char:
                # Found a match! Return the canonical entity name.
                log_debug(f"Span ({span_start_char}, {span_end_char}) matched mention '{mention['text']}' -> Entity '{canonical_entity}'")
                return canonical_entity
    # If no match was found after checking all mentions for all entities
    log_debug(f"Span ({span_start_char}, {span_end_char}) did not match any coref mention.")
    return None

# --- Existing Entity Extraction ---
def extract_entities(text: str) -> Dict[str, str]:
    """Extract named entities with spaCy. Returns {entity_text: entity_label}."""
    nlp = get_spacy_model()
    if nlp is None:
        log_error("spaCy model not initialized. Cannot extract entities.")
        return {}
    try:
        doc = nlp(text)
        return {ent.text.strip(): ent.label_ for ent in doc.ents}
    except Exception as e:
        log_error(f"Error during spaCy entity extraction: {e}", exc_info=True)
        return {}

# --- Existing Structured Coreference Resolution ---
def resolve_coreferences_structured(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Resolves coreferences using FastCoref and correlates with spaCy NER.
    Uses spaCy for initial tokenization to align indices.
    Improved logic for linking clusters to NER entities, checking exact and root matches.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary where keys are canonical
        named entity strings, and values are lists of mention dictionaries.
        Each mention dictionary contains 'text', 'start_char', 'end_char'.
        Example: {'Cinderella': [{'text': 'She', 'start_char': 10, 'end_char': 13}, ...]}
    """
    # 1. Get Models
    nlp = get_spacy_model()
    fc_model = get_fastcoref_model()

    if nlp is None or fc_model is None:
        log_error("spaCy or FastCoref model not initialized. Cannot resolve coreferences.")
        return {}

    if not text.strip():
        log_debug("Received empty text for structured coreference resolution.")
        return {}

    log_info(f"Starting structured coreference resolution for text: '{text[:100]}...'")

    try:
        # 2. Run spaCy NER and Tokenization
        doc = nlp(text)
        ner_entities_by_start_char: Dict[int, Span] = {ent.start_char: ent for ent in doc.ents}
        canonical_ner_texts = {ent.text.strip() for ent in doc.ents}
        log_debug(f"Found {len(canonical_ner_texts)} canonical NER entities: {canonical_ner_texts}")

        spacy_tokens = [token.text for token in doc]
        log_debug(f"spaCy tokenized text into {len(spacy_tokens)} tokens.")
        if not spacy_tokens:
             log_warning("spaCy tokenization resulted in zero tokens.")
             return {}

        # 3. Run FastCoref with pre-tokenized input
        log_debug("Running FastCoref prediction with spaCy tokens...")
        preds = fc_model.predict(texts=[spacy_tokens], is_split_into_words=True)
        log_debug("FastCoref prediction completed.")

        if not preds or not preds[0]:
            log_info("FastCoref returned no predictions.")
            return {}

        clusters_token_indices = preds[0].get_clusters(as_strings=False)
        log_info(f"FastCoref found {len(clusters_token_indices)} clusters (token indices).")

        # 4. Process Clusters and Map to NER Entities
        resolved_coreferences: Dict[str, List[Dict[str, Any]]] = {}

        for cluster in clusters_token_indices:
            if not cluster: continue

            cluster_mentions_data = []
            canonical_entity_for_cluster: Optional[str] = None
            best_match_info: Tuple[int, int, Optional[str]] = (2, -1, None)

            for mention_token_indices in cluster:
                start_token_idx, end_token_idx = mention_token_indices

                if start_token_idx < 0 or end_token_idx >= len(doc):
                    log_warning(f"Cluster mention token indices {mention_token_indices} out of bounds for doc length {len(doc)}. Skipping mention.")
                    continue

                try:
                    mention_span = doc[start_token_idx : end_token_idx + 1]
                except IndexError:
                     log_warning(f"IndexError accessing doc[{start_token_idx}:{end_token_idx + 1}] for doc length {len(doc)}. Skipping mention.")
                     continue

                mention_text = mention_span.text.strip()
                mention_start_char = mention_span.start_char
                mention_end_char = mention_span.end_char

                if not mention_text:
                    log_debug(f"Empty mention text for token indices {mention_token_indices}. Skipping.")
                    continue

                mention_data = {
                    "text": mention_text,
                    "start_char": mention_start_char,
                    "end_char": mention_end_char
                }
                cluster_mentions_data.append(mention_data)

                current_match_type = 2
                current_match_text = None
                current_match_len = -1

                if mention_start_char in ner_entities_by_start_char:
                     ner_span = ner_entities_by_start_char[mention_start_char]
                     if ner_span.end_char == mention_end_char and ner_span.text.strip() == mention_text:
                          current_match_type = 0
                          current_match_text = ner_span.text.strip()
                          current_match_len = len(current_match_text)
                          log_debug(f"Mention '{mention_text}' is an exact NER match for '{current_match_text}'")

                if current_match_type > 0:
                    root_token = mention_span.root
                    root_text = root_token.text.strip()
                    if root_text in canonical_ner_texts:
                        current_match_type = 1
                        current_match_text = root_text
                        current_match_len = len(current_match_text)
                        log_debug(f"Mention '{mention_text}' has root '{root_text}' matching NER entity '{current_match_text}'")

                if current_match_text is not None:
                    if current_match_type < best_match_info[0] or \
                       (current_match_type == best_match_info[0] and current_match_len > best_match_info[1]):
                        best_match_info = (current_match_type, current_match_len, current_match_text)
                        log_debug(f"Updating best anchor for cluster to '{current_match_text}' (type: {current_match_type}, len: {current_match_len})")

            canonical_entity_for_cluster = best_match_info[2]

            if canonical_entity_for_cluster:
                if canonical_entity_for_cluster not in resolved_coreferences:
                    resolved_coreferences[canonical_entity_for_cluster] = []
                resolved_coreferences[canonical_entity_for_cluster].extend(cluster_mentions_data)
                log_debug(f"Linked cluster to canonical entity '{canonical_entity_for_cluster}'. Added {len(cluster_mentions_data)} mentions.")
            else:
                 if cluster_mentions_data:
                      log_debug(f"Could not link cluster with mentions {[m['text'] for m in cluster_mentions_data]} to a canonical NER entity. Discarding cluster.")

        for entity, mentions in resolved_coreferences.items():
             unique_mentions = []
             seen_spans = set()
             for mention in mentions:
                  span_tuple = (mention['start_char'], mention['end_char'])
                  if span_tuple not in seen_spans:
                       unique_mentions.append(mention)
                       seen_spans.add(span_tuple)
             resolved_coreferences[entity] = unique_mentions
             if len(unique_mentions) < len(mentions):
                 log_debug(f"Deduplicated mentions for entity '{entity}'.")

        log_info(f"Structured coreference resolution successful. Found mappings for {len(resolved_coreferences)} entities.")
        return resolved_coreferences

    except Exception as e:
        log_error(f"Error during structured coreference resolution: {e}", exc_info=True)
        return {}

RE_MODEL_NAME = "Babelscape/rebel-large" # Using REBEL model

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

# --- Helper Function for REBEL Output Parsing ---
def extract_triplets_rebel(text: str) -> List[Dict[str, str]]:
    """
    Parses the generated text from the REBEL model to extract structured triplets.
    Based on the function provided in the REBEL model card.
    """
    triplets = []
    relation, subject, obj = '', '', '' # Changed object_ to obj to avoid clash
    text = text.strip()
    current = 'x'
    # Handle potential variations in special tokens (e.g., with spaces)
    # Split carefully to handle multiple spaces between tokens potentially
    tokens = re.split(r'\s+', text.replace("<s>", "").replace("<pad>", "").replace("</s>", ""))

    for token in tokens:
        if not token: # Skip empty tokens resulting from split
            continue

        if token == "<triplet>":
            current = 't'
            if relation: # Save previous triplet if complete
                # Clean extra spaces accumulated during concatenation
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': obj.strip()})
                relation = '' # Reset relation for the new triplet
            subject = '' # Reset subject for the new triplet
        elif token == "<subj>":
            current = 's'
            if relation: # Save previous triplet if complete (subj follows obj)
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': obj.strip()})
                # Don't reset subject here, a new relation/object follows
            obj = '' # Reset object for the new entity pair
            relation = '' # Reset relation
        elif token == "<obj>":
            current = 'o'
            relation = '' # Reset relation, object follows
        else:
            # Append token to the current part (subject, object, or relation)
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                obj += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    # Capture the last triplet in the sequence if fully formed
    if subject and relation and obj:
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': obj.strip()})

    return triplets

# --- Relation Extraction Function ---
def extract_relations(
    text: str,
    resolved_coreferences: Dict[str, List[Dict[str, Any]]],
    re_model_name: str = RE_MODEL_NAME
) -> List[Dict[str, Any]]:
    """
    Extracts relationships between entities in a text based on coreferences,
    using the REBEL model for end-to-end relation extraction.

    Args:
        text: The original text document.
        resolved_coreferences: A dictionary where keys are canonical entity names
                               and values are lists of mention dictionaries.
                               (Used primarily to identify sentences containing
                               multiple known entities).
        re_model_name: The name of the Hugging Face model to use.

    Returns:
        A list of dictionaries, each representing an extracted relationship triplet
        as identified by REBEL within relevant sentences. Keys are 'sentence',
        'sentence_span', 'head' (subject text), 'type' (relation text),
        'tail' (object text).
    """
    print(f"\n--- Starting Relation Extraction for text snippet ---")
    print(f"Using RE model: {re_model_name}")

    # --- Model Loading ---
    model_loaded_locally = False
    local_model = None
    local_tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # REBEL might work better on CPU if GPU memory is limited, but let's try GPU first if available
    print(f"Using device: {device}")

    try:
        print(f"Loading Tokenizer and Model: {re_model_name}...")
        # Load Seq2Seq model and tokenizer
        local_tokenizer = AutoTokenizer.from_pretrained(re_model_name)
        local_model = AutoModelForSeq2SeqLM.from_pretrained(re_model_name)
        local_model.to(device)
        local_model.eval()
        print("Model and Tokenizer loaded successfully.")
        model_loaded_locally = True
    except Exception as e:
        print(f"Error loading model or tokenizer '{re_model_name}': {e}")
        print("Model loading failed. Cannot perform relation extraction.")
        return []


    # --- Inference Function (using REBEL) ---
    def get_relations_from_sentence_rebel(model, tokenizer, sentence):
        """
        Performs inference using the REBEL model on a single sentence
        and parses the output to extract triplets.
        """
        print(f"\n--- Analyzing Sentence with REBEL: '{sentence.strip()}' ---")

        if not (model and tokenizer):
             print("--- (Model not loaded, skipping relation prediction) ---")
             return [] # Return empty list if model isn't loaded

        try:
            # Tokenize the sentence
            # Note: REBEL documentation uses default padding/truncation settings.
            # Adjust max_length if needed for very long sentences.
            model_inputs = tokenizer(
                sentence,
                max_length=256, # Default from docs, adjust if needed
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Generation arguments (from REBEL docs, can be customized)
            gen_kwargs = {
                "max_length": 256, # Max length of the generated output sequence
                "length_penalty": 0, # No penalty
                "num_beams": 3, # Beam search width
                "num_return_sequences": 1, # Generate one best sequence
            }

            # Generate token IDs
            print("--- (Generating relation text with REBEL...) ---")
            with torch.no_grad():
                generated_tokens = model.generate(
                    model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    **gen_kwargs,
                )

            # Decode generated tokens into text, keeping special tokens for parsing
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

            # Parse the decoded text to extract triplets
            all_triplets = []
            for sentence_pred in decoded_preds:
                print(f"--- Generated Text: {sentence_pred} ---")
                extracted = extract_triplets_rebel(sentence_pred)
                print(f"--- Extracted Triplets: {extracted} ---")
                all_triplets.extend(extracted)

            return all_triplets

        except Exception as e:
            import traceback
            print(f"--- Error during REBEL inference or parsing: {e} ---")
            # traceback.print_exc() # Uncomment for full stack trace
            return [] # Return empty list on error


    # --- Preprocessing (Identifies sentences with co-occurring coref entities) ---

    # 1. Build mention_to_canonical map (handling ambiguities)
    mention_to_canonical: Dict[Tuple[int, int], Dict[str, str]] = {}
    canonical_to_mentions: Dict[str, List[Dict[str, Any]]] = resolved_coreferences
    mention_counts = defaultdict(int)
    mentions_to_skip = set()
    for canonical, mentions in canonical_to_mentions.items():
        for mention in mentions:
            start = mention.get('start_char')
            end = mention.get('end_char')
            if start is not None and end is not None:
                span = (start, end)
                mention_counts[span] += 1
                if mention_counts[span] > 1:
                    mentions_to_skip.add(span)
            else:
                print(f"Warning: Mention missing start/end char in {canonical}: {mention}")

    for canonical, mentions in canonical_to_mentions.items():
        for mention in mentions:
            start = mention.get('start_char')
            end = mention.get('end_char')
            m_text = mention.get('text', '[text missing]')
            if start is not None and end is not None:
                span = (start, end)
                if span not in mentions_to_skip:
                    mention_to_canonical[span] = {"canonical": canonical, "text": m_text}

    # 2. Sentence Splitting
    sentence_spans: List[Dict[str, Any]] = []
    start_offset = 0
    try:
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            try:
                current_start = text.index(sentence, start_offset)
                current_end = current_start + len(sentence)
                sentence_spans.append({
                    "text": sentence,
                    "start_char": current_start,
                    "end_char": current_end
                })
                start_offset = current_end
            except ValueError:
                print(f"Warning: Could not precisely locate sentence in original text: '{sentence[:50]}...'")
                approx_end = start_offset + len(sentence)
                sentence_spans.append({
                    "text": sentence,
                    "start_char": start_offset,
                    "end_char": approx_end
                })
                start_offset = approx_end + 1

    except Exception as e:
        print(f"Error during sentence tokenization: {e}")
        return []

    # --- Relation Extraction Logic ---
    extracted_relationships: List[Dict[str, Any]] = []

    # 3. Iterate through sentences
    # We only process sentences containing at least two *different* canonical entities
    # according to the coreference map, as a heuristic to focus REBEL.
    # REBEL itself doesn't need the entity pair beforehand, but this filters sentences.
    processed_sentences = set() # Avoid processing the same sentence multiple times if entities overlap spans
    for sentence_info in sentence_spans:
        sentence_text = sentence_info["text"]
        sent_start = sentence_info["start_char"]
        sent_end = sentence_info["end_char"]

        # Check if this sentence span has already been processed
        sentence_span_tuple = (sent_start, sent_end)
        if sentence_span_tuple in processed_sentences:
            continue

        entities_in_sentence_canons = set()
        # Find all canonical entities mentioned within this sentence's span
        for mention_span, entity_info in mention_to_canonical.items():
            mention_start, mention_end = mention_span
            if sent_start <= mention_start < sent_end and sent_start < mention_end <= sent_end:
                 entities_in_sentence_canons.add(entity_info["canonical"])

        # If sentence contains mentions of at least two distinct canonical entities from coref map...
        if len(entities_in_sentence_canons) >= 2:
            # ...then run REBEL on this sentence.
            sentence_triplets = get_relations_from_sentence_rebel(
                local_model,
                local_tokenizer,
                sentence_text
            )

            # Add context (sentence, span) to the extracted triplets
            for triplet in sentence_triplets:
                extracted_relationships.append({
                    "sentence": sentence_text.strip(),
                    "sentence_span": sentence_span_tuple,
                    "head": triplet['head'],
                    "type": triplet['type'],
                    "tail": triplet['tail']
                    # Note: We don't include confidence score as REBEL doesn't directly provide one per triplet.
                })
            # Mark this sentence span as processed
            processed_sentences.add(sentence_span_tuple)


    print(f"--- Relation Extraction Finished. Found {len(extracted_relationships)} potential relationships using REBEL. ---")
    return extracted_relationships

def format_graph_to_text(graph_data: List[Dict[str, Any]]) -> str:
    """
    Formats the graph data retrieved from Neo4j into a readable text string.

    Args:
        graph_data: A list of dictionaries, where each dictionary represents a node
                    and its connections, ordered by importance. Expected keys:
                    'node_properties', 'mention_count', 'outgoing_rels', 'incoming_rels'.

    Returns:
        A formatted string representing the graph.
    """
    if not graph_data:
        return "No graph data found for this book."

    output_lines = []
    output_lines.append("--- Knowledge Graph Summary ---")
    output_lines.append(f"Book Title: {graph_data[0]['node_properties'].get('book_title', 'Unknown')}\n") # Assumes all nodes have same title

    for i, node_data in enumerate(graph_data):
        props = node_data.get("node_properties", {})
        mention_count = node_data.get("mention_count", 0)
        outgoing = node_data.get("outgoing_rels", [])
        incoming = node_data.get("incoming_rels", [])

        name = props.get('name', 'Unnamed Node')
        node_type = props.get('type', 'Unknown Type')

        # --- Node Header ---
        output_lines.append(f"--- {i+1}. {name} ({node_type}) ---")
        output_lines.append(f"   Importance (Mentions): {mention_count}")
        if 'mentions' in props and props['mentions']:
             output_lines.append(f"   Mentioned in Sections: {sorted(list(set(props['mentions'])))}") # Show unique sorted sections

        # --- Node Properties ---
        output_lines.append("   Properties:")
        prop_count = 0
        for key, value in props.items():
            # Skip core identifiers already displayed
            if key not in ['name', 'type', 'book_title', 'mentions']:
                output_lines.append(f"     - {key}: {value}")
                prop_count += 1
        if prop_count == 0:
             output_lines.append("     (No additional properties)")


        # --- Relationships ---
        output_lines.append("   Relationships:")
        rel_count = 0
        # Outgoing
        if outgoing:
            for rel in outgoing:
                target_props = rel.get('target_node_properties', {})
                target_name = target_props.get('name', 'Unknown Node')
                rel_type = rel.get('rel_type', 'UNKNOWN_REL')
                output_lines.append(f"     - [{rel_type}]-> ({target_name})")
                rel_count += 1
        # Incoming
        if incoming:
             for rel in incoming:
                source_props = rel.get('source_node_properties', {})
                source_name = source_props.get('name', 'Unknown Node')
                rel_type = rel.get('rel_type', 'UNKNOWN_REL')
                output_lines.append(f"     <-[{rel_type}]- ({source_name})")
                rel_count += 1

        if rel_count == 0:
             output_lines.append("     (No relationships within this book)")

        output_lines.append("") # Add a blank line between nodes

    return "\n".join(output_lines)
