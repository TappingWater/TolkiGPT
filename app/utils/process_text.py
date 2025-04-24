import spacy
from fastcoref import FCoref
from typing import Dict, List, Tuple, Optional, Any
import re
import logging
from app.db import execute_query # Assuming your Neo4j utils are in app.db
from spacy.tokens import Token, Span, Doc

# --- Configuration ---
SPACY_MODEL_NAME = "en_core_web_lg"
FASTCOREF_MODEL_NAME = "biu-nlp/f-coref" 

# --- Logging ---
# Ensure logging is configured elsewhere or here
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Define the Class ---
class TextProcessor:
    # --- Load Models as Class Attributes ---
    try:
        log.info(f"Loading spaCy model '{SPACY_MODEL_NAME}'...")
        # Assign to class attribute 'nlp'
        nlp: spacy.Language = spacy.load(SPACY_MODEL_NAME)
        log.info("spaCy model loaded.")

        log.info(f"Loading FastCoref model '{FASTCOREF_MODEL_NAME}'...")
        # Assign to class attribute 'fc_model'
        # Consider device='cuda' if GPU is available and configured
        fc_model = FCoref(model_name_or_path=FASTCOREF_MODEL_NAME, device='cpu')
        log.info("FastCoref model loaded.")
        MODELS_LOADED = True
    except ImportError as ie:
         log.exception(f"FATAL: Missing libraries. spaCy or fastcoref not installed properly? {ie}")
         MODELS_LOADED = False
         # Depending on your app, you might raise an error here to stop startup
         # raise RuntimeError("Failed to initialize TextProcessor models due to missing libraries") from ie
    except OSError as e:
        log.exception(f"FATAL: Failed to load NLP models (likely not downloaded): {e}")
        log.error(f"Ensure models are downloaded (e.g., python -m spacy download {SPACY_MODEL_NAME})")
        MODELS_LOADED = False
        # raise RuntimeError("Failed to initialize TextProcessor models") from e
    except Exception as e:
        log.exception(f"FATAL: An unexpected error occurred loading NLP models: {e}")
        MODELS_LOADED = False
        # raise RuntimeError("Failed to initialize TextProcessor models") from e

    # --- Methods now access models via TextProcessor.nlp or TextProcessor.fc_model ---

    @staticmethod
    def extract_entities(text: str) -> Dict[str, str]:
        """Extract named entities with spaCy"""
        if not TextProcessor.MODELS_LOADED:
             log.error("Cannot extract entities, models not loaded.")
             return {}
        # Access class attribute nlp
        doc = TextProcessor.nlp(text)
        return {ent.text.strip(): ent.label_ for ent in doc.ents}

    @staticmethod
    def resolve_coreferences(text: str) -> Dict[str, str]:
        """Resolve coreferences with FastCoref"""
        if not TextProcessor.MODELS_LOADED:
            log.error("Cannot resolve coreferences, models not loaded.")
            return {}
        try:
            if not text.strip(): return {}
            # Access class attribute fc_model
            preds = TextProcessor.fc_model.predict(texts=[text])
            if not preds or not preds[0]: return {}
            coref_map = {}
            clusters = preds[0].get_clusters(as_strings=True)
            # log.info(f"FastCoref found {len(clusters)} clusters.") # Reduce log noise maybe
            for cluster in clusters:
                if not cluster: continue
                main_entity = cluster[0]
                for mention in cluster:
                    coref_map[mention] = main_entity
            return coref_map
        except Exception as e:
            log.error(f"FastCoref prediction failed: {e}")
            return {}

    @staticmethod
    def _get_entity_span_text(token: Token, doc: Doc, coref_map: Dict[str, str]) -> Optional[str]:
        """Helper to get resolved entity text (uses doc, no direct model access needed)."""
        # This method doesn't directly call nlp or fc_model, uses the passed 'doc'
        resolved_text = None
        original_text = None
        entity_span = next((ent for ent in doc.ents if ent.start <= token.i < ent.end), None)
        if entity_span:
            original_text = entity_span.text.strip()
        else:
            subtree_span = doc[token.left_edge.i : token.right_edge.i + 1]
            original_text = subtree_span.text.strip()
        if not original_text: return None
        resolved_text = coref_map.get(original_text, original_text)
        # Optional logging moved to extract_relations for context
        # if resolved_text != original_text: log.info(f"COREFERENCE APPLIED: '{original_text}' -> '{resolved_text}'")
        return resolved_text

    @staticmethod
    def extract_relations(
        text: str,
        coref_map: Dict[str, str],
        entities: Dict[str, str]
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Extract filtered relationships."""
        if not TextProcessor.MODELS_LOADED:
            log.error("Cannot extract relations, models not loaded.")
            return []

        # Access class attribute nlp
        doc = TextProcessor.nlp(text)
        relations = []
        # log.info(f"Starting relation extraction...") # Reduce log noise

        # --- Relation extraction logic ---
        for token in doc:
            if token.pos_ != "VERB": continue

            subj_token, obj_token, prep_for_rel = None, None, None
            rel_base = token.lemma_.upper()

            # Passive check
            passive_subj = next((c for c in token.children if c.dep_ == "nsubjpass"), None)
            if passive_subj:
                obj_token = passive_subj
                agent = next((c for c in token.children if c.dep_ == "agent"), None)
                if agent: subj_token = next((gc for gc in agent.children if gc.dep_ == "pobj"), None)

            # Active check (if passive incomplete)
            if not subj_token or not obj_token:
                active_subj = next((c for c in token.children if c.dep_ == "nsubj"), None)
                if active_subj:
                    if not subj_token: subj_token = active_subj
                    # Find Object (prioritized)
                    dobj = next((c for c in token.children if c.dep_ == "dobj"), None)
                    pobj_info = next(((gc, p.text.upper()) for p in token.children if p.dep_ == "prep" for gc in p.children if gc.dep_ == "pobj"), None)
                    dative = next((c for c in token.children if c.dep_ == "dative"), None)
                    temp_obj_token = None
                    if dobj: temp_obj_token = dobj
                    elif pobj_info: temp_obj_token, prep_for_rel = pobj_info
                    elif dative: temp_obj_token = dative
                    if not obj_token and temp_obj_token: obj_token = temp_obj_token

            # Resolve, Filter, Store
            if subj_token and obj_token:
                subj_resolved = TextProcessor._get_entity_span_text(subj_token, doc, coref_map)
                obj_resolved = TextProcessor._get_entity_span_text(obj_token, doc, coref_map)

                if not subj_resolved or not obj_resolved: continue
                if subj_resolved == obj_resolved: continue

                # Filtering Logic
                subj_is_ner = subj_resolved in entities
                obj_is_ner = obj_resolved in entities

                if subj_is_ner or obj_is_ner:
                    # Log resolution if it happened for passed relations
                    subj_orig = doc[subj_token.left_edge.i : subj_token.right_edge.i + 1].text.strip()
                    obj_orig = doc[obj_token.left_edge.i : obj_token.right_edge.i + 1].text.strip()
                    log_coref = ""
                    if subj_resolved != subj_orig: log_coref += f" Subj: '{subj_orig}'->'{subj_resolved}'"
                    if obj_resolved != obj_orig: log_coref += f" Obj: '{obj_orig}'->'{obj_resolved}'"

                    rel_type = f"{rel_base}_{prep_for_rel}" if prep_for_rel else rel_base
                    attrs = {} # Attribute extraction logic...
                    for child in token.children:
                        if child.dep_ == "prep":
                            prep_obj = next((gc for gc in child.children if gc.dep_ == "pobj"), None)
                            if prep_obj:
                                ent_span = next((ent for ent in doc.ents if ent.start <= prep_obj.i < ent.end), None)
                                if ent_span:
                                     if ent_span.label_ in ["DATE", "TIME"]: attrs["time"] = ent_span.text.strip()
                                     elif ent_span.label_ in ["GPE", "LOC"]: attrs["location"] = ent_span.text.strip()

                    log.info(f"ADDING RELATION (Passed Filter): ('{subj_resolved}', '{rel_type}', '{obj_resolved}', {attrs}){log_coref}")
                    relations.append((subj_resolved, rel_type, obj_resolved, attrs))
                # else: # Log filtered out relations if needed (can be noisy)
                #    log.debug(f"Relation FILTERED OUT (Neither endpoint NER): ('{subj_resolved}', '{obj_resolved}')")

        return relations

    # --- generate_upsert_queries (No changes needed) ---
    @staticmethod
    def generate_upsert_queries(
        entities: Dict[str, str],
        relationships: List[Tuple[str, str, str, Dict[str, Any]]],
        additional_attributes: Optional[Dict[str, Any]] = None,
        book_title: Optional[str] = None,
        book_section: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
         # ... (Same implementation as the previous answer) ...
        queries = []
        base_attrs = additional_attributes or {}
        node_names = set(entities.keys())
        for subj, _, obj, _ in relationships:
            node_names.add(subj)
            node_names.add(obj)
        for name in node_names:
            if not name: continue
            label = entities.get(name, "Concept")
            safe_label = f"`{label}`" if not label.isalnum() else label
            node_attrs = base_attrs.copy(); query_params = {"name": name}
            if book_title: node_attrs['book_title'] = book_title
            query_params["node_attrs"] = node_attrs
            mention_update_clause = ""
            if book_section:
                query_params["book_section"] = book_section
                mention_update_clause = "n.mentions = CASE WHEN n.mentions IS NULL THEN [$book_section] WHEN NOT $book_section IN n.mentions THEN n.mentions + $book_section ELSE n.mentions END"
            queries.append((f"""MERGE (n:{safe_label} {{name: $name}}) ON CREATE SET n+= $node_attrs, n.created_at = datetime(), n._source='text_import' {', '+mention_update_clause if mention_update_clause else ''} ON MATCH SET n.last_updated=datetime(), n+= $node_attrs {', '+mention_update_clause if mention_update_clause else ''}""",query_params))
        for subj, rel, obj, attrs in relationships:
             if not subj or not obj or not rel: continue
             safe_rel = re.sub(r'[^a-zA-Z0-9_]', '_', rel).upper()[:60]
             if not safe_rel: continue
             rel_attrs = attrs.copy(); rel_attrs.update(base_attrs)
             if book_title: rel_attrs['book_title'] = book_title
             if book_section: rel_attrs['book_section'] = book_section
             query_params = {"subj": subj, "obj": obj, "attrs": rel_attrs}
             on_match_updates = []
             if book_section: query_params["book_section"] = book_section; on_match_updates.append("r.book_section = $book_section")
             if book_title: query_params["book_title"] = book_title; on_match_updates.append("r.book_title = $book_title")
             if "time" in attrs: query_params["time"] = attrs["time"]; on_match_updates.append("r.time = $time")
             if "location" in attrs: query_params["location"] = attrs["location"]; on_match_updates.append("r.location = $location")
             on_match_clause = ", ".join(on_match_updates)
             queries.append((f"""MATCH (a {{name: $subj}}), (b {{name: $obj}}) MERGE (a)-[r:`{safe_rel}`]->(b) ON CREATE SET r = $attrs, r.created_at = datetime(), r._source = 'text_import' ON MATCH SET r += $attrs, r.last_updated = datetime() {', '+on_match_clause if on_match_clause else ''}""", query_params))
        return queries


    # --- process_text_to_neo4j (No structural changes needed, uses class methods) ---
    @classmethod
    def process_text_to_neo4j(
        cls,
        text: str,
        database: str = "neo4j",
        additional_attributes: Optional[Dict[str, Any]] = None,
        book_title: Optional[str] = None,
        book_section: Optional[str] = None
    ) -> bool:
        """Complete text processing pipeline."""
        if not cls.MODELS_LOADED:
             log.error("Cannot process text, models failed to load during initialization.")
             return False
        if not text or not text.strip():
            log.warning("Input text is empty or whitespace only. Skipping processing.")
            return False

        try:
            # Calls now correctly use methods that access class attributes for models
            entities = cls.extract_entities(text)
            log.info(f"Extracted {len(entities)} named entities: {list(entities.keys())}")

            coref_map = cls.resolve_coreferences(text)
            log.info(f"Resolved {len(coref_map)} coreference mentions.")

            relations = cls.extract_relations(text, coref_map, entities)
            log.info(f"Found {len(relations)} relations AFTER filtering.")

            queries = cls.generate_upsert_queries(
                entities, relations, additional_attributes, book_title, book_section
            )
            log.info(f"Generated {len(queries)} Cypher queries for upsert.")

            # Query execution loop
            success_count, failure_count = 0, 0
            for i, (query, params) in enumerate(queries):
                if execute_query(query, params, database):
                    success_count += 1
                else:
                    failure_count += 1
                    log.error(f"Failed executing Query {i+1}/{len(queries)}") # Params can be large, logged elsewhere if needed

            log.info(f"Query execution completed. Success: {success_count}, Failed: {failure_count}.")
            return failure_count == 0

        except Exception as e:
            # Log the exception that occurs *during* processing
            log.exception(f"Processing failed unexpectedly in process_text_to_neo4j: {str(e)}")
            return False

# --- End of TextProcessor Class ---