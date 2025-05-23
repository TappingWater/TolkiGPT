import spacy
from fastcoref import FCoref
import subprocess
import sys
from typing import Any, Optional, Tuple
from app.utils import log_info,log_error, log_debug, log_warning
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import app.config as config
from peft import PeftModel

_spacy_nlp: Optional[spacy.Language] = None
_coref_model: Optional[FCoref] = None
_gen_tokenizer: Optional[Any] = None 
_gen_model: Optional[Any] = None
_models_initialized: bool = False 

_spacy_nlp: Optional[spacy.Language] = None
_coref_model: Optional[FCoref] = None
_gen_tokenizer: Optional[Any] = None
_gen_model: Optional[Any] = None
_models_initialized: bool = False

def _load_gen_model(
    base_model_name_or_path: str,
    adapter_name_or_path: str
) -> Tuple[Optional[Any], Optional[Any]]:
    log_info(f"Attempting to load generative model: Base='{base_model_name_or_path}', Adapter='{adapter_name_or_path}'")
    try:
        log_debug(f"Loading base model '{base_model_name_or_path}'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float16, 
            device_map=config.DEFAULT_INFERENCE_DEVICE,       
            trust_remote_code=True    
        )
        log_info(f"Base model '{base_model_name_or_path}' loaded.")
        log_debug(f"Loading tokenizer from adapter location '{adapter_name_or_path}'...")
        # Load tokenizer from the adapter path/ID - it might contain specific tokens/config
        tokenizer = AutoTokenizer.from_pretrained(adapter_name_or_path)
        # Set pad token if it's not set (common issue)
        if tokenizer.pad_token is None:
            log_warning("Tokenizer pad_token not set. Using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # If you modified the base model's tokenizer during finetuning,
            # you might need to resize embeddings:
            # base_model.resize_token_embeddings(len(tokenizer))
        log_info(f"Tokenizer loaded from '{adapter_name_or_path}'.")
        log_debug(f"Applying PEFT adapter '{adapter_name_or_path}' to base model...")
        # Load the PEFT model - this applies the adapter weights to the base model
        model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
        model.eval() # Set the combined model to evaluation mode
        log_info(f"PEFT adapter '{adapter_name_or_path}' applied successfully.")
        # No need for model.to() if device_map="auto" was used
        log_info(f"Generative model ready on device(s): {model.device if hasattr(model, 'device') else 'Multiple (device_map used)'}")
        return tokenizer, model

    except ImportError as ie:
         log_error(f"Missing libraries for PEFT/Transformers. Ensure 'peft', 'transformers', 'torch', 'accelerate', 'bitsandbytes' are installed. Error: {ie}", exc_info=False)
         return None, None
    except Exception as e:
        log_error(f"Failed to load generative model (Base: '{base_model_name_or_path}', Adapter: '{adapter_name_or_path}').", exc_info=True)
        return None, None

def _load_spacy_model(model_name: str) -> Optional[spacy.Language]:
    """
    Internal helper to load/download spaCy model.
    Logs errors but doesn't re-raise all exceptions to allow partial loading.
    """
    try:
        log_info(f"Attempting to load spaCy model '{model_name}'...")
        nlp = spacy.load(model_name)
        log_info(f"spaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError:
        log_warning(f"spaCy model '{model_name}' not found locally. Attempting download...")
        try:
            log_debug(f"Running command: {[sys.executable, '-m', 'spacy', 'download', model_name]}")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            log_info(f"spaCy model '{model_name}' downloaded successfully.")
            log_info(f"Attempting to load spaCy model '{model_name}' again after download...")
            nlp = spacy.load(model_name)
            log_info(f"spaCy model '{model_name}' loaded successfully after download.")
            return nlp
        except subprocess.CalledProcessError as cpe:
            error_message = cpe.stderr.decode().strip() if cpe.stderr else "No error output captured."
            log_error(f"Failed to download spaCy model '{model_name}'. Error: {error_message}", exc_info=False)
            return None
        except OSError:
            log_error(f"Failed to load spaCy model '{model_name}' even after attempting download.", exc_info=True)
            return None
        except Exception:
            log_error(f"An unexpected error occurred during spaCy model download/load for '{model_name}'.", exc_info=True)
            return None
    except ImportError:
        log_error("spaCy library not found. Please install it (`pip install spacy`).")
        return None
    except Exception:
         log_error(f"An unexpected error occurred while trying to load spaCy model '{model_name}'.", exc_info=True)
         return None

def _load_coref_model(model_name_or_path: str, device: str) -> Optional[FCoref]:
    """
    Internal helper to load/download Coreference model
    Logs errors but doesn't re-raise all exceptions.
    """
    try:
        log_info(f"Attempting to load FastCoref model '{model_name_or_path}' on device '{device}'...")
        fc_model = FCoref(model_name_or_path=model_name_or_path, device=device)
        log_info(f"FastCoref model '{model_name_or_path}' loaded successfully.")
        return fc_model
    except ImportError as ie:
        log_error(f"Missing libraries required for FastCoref ('transformers', 'torch'/'tensorflow'). Error: {ie}", exc_info=False)
        return None
    except Exception as e:
        log_error(f"Failed to load or download FastCoref model '{model_name_or_path}'.", exc_info=True)
        return None

# --- Initialization Function ---
def initialize_models(
    spacy_model_name,
    fastcoref_model_name,
    gen_base_model_name,
    gen_adapter_name,
    device
):
    """
    Loads both models and stores them in module-level variables.
    This function is called automatically when the module is imported.
    """
    global _spacy_nlp, _coref_model, _gen_tokenizer, _gen_model, _models_initialized
    if _models_initialized:
        log_debug("Models already initialized.")
        return
    log_info("--- Initializing NLP models ---")
    _spacy_nlp = _load_spacy_model(spacy_model_name)
    _coref_model = _load_coref_model(fastcoref_model_name, device=device)
    _gen_tokenizer, _gen_model = _load_gen_model(
        base_model_name_or_path=gen_base_model_name,
        adapter_name_or_path=gen_adapter_name,
    )
    _models_initialized = True 
    
    # Log the overall outcome
    loaded_models = []
    failed_models = []
    if _spacy_nlp: 
        loaded_models.append("spaCy")
    else: 
        failed_models.append("spaCy")
    if _coref_model: 
        loaded_models.append("FastCoref")
    else: 
        failed_models.append("FastCoref")
    if _gen_tokenizer and _gen_model: 
        loaded_models.append("Generative")
    else: 
        failed_models.append("Generative")
    if not failed_models:
        log_info(f"--- All models initialized successfully: {', '.join(loaded_models)}. ---")
    else:
        log_warning(f"--- Model initialization complete. Loaded: {', '.join(loaded_models) or 'None'}. Failed: {', '.join(failed_models)}. ---")

# --- Getter Functions ---
def get_spacy_model() -> Optional[spacy.Language]:
    return _spacy_nlp

def get_fastcoref_model() -> Optional[FCoref]:
    return _coref_model

def get_gen_tokenizer() -> Optional[Any]:
    """Returns the loaded generative model tokenizer instance."""
    return _gen_tokenizer

def get_gen_model() -> Optional[Any]:
    """Returns the loaded generative model instance."""
    return _gen_model
