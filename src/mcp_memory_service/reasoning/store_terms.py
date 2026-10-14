"""Store-specific terms extractor for domain entities."""

import json
import re
import logging
from pathlib import Path
from .entities import Entity

logger = logging.getLogger(__name__)

_STORE_TERMS_PATH = Path(__file__).parent.parent / 'data' / 'store_terms.json'


class StoreTermsExtractor:
    """Extract domain terms specific to stores (mir, rer, mcr, etc)."""
    
    def __init__(self, path=None):
        """Initialize extractor with optional custom path."""
        self._path = Path(path) if path else _STORE_TERMS_PATH
        self._data = self._load()
        self._compiled = self._compile(self._data)
    
    def _load(self):
        """Load store terms from JSON file, returning empty dict on error."""
        try:
            if not self._path.exists():
                logger.warning(f"Store terms file not found: {self._path}")
                return {}
            
            with open(self._path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in store terms file {self._path}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Error loading store terms file {self._path}: {e}")
            return {}
    
    def _compile(self, data):
        """Compile regex patterns for each store's terms."""
        compiled = {}
        
        for store, cfg in data.items():
            terms = []
            if isinstance(cfg, dict) and 'terms' in cfg:
                terms = cfg['terms']
            elif isinstance(cfg, list):
                terms = cfg
            
            compiled_terms = []
            for term in terms:
                if term and term.strip():  # Skip empty/whitespace terms
                    # Word boundary regex to prevent false positives.
                    # Matches entities.py:88 (MCP_ENTITY_CUSTOM_TERMS) for
                    # consistency across the project: '-' and '_' count as part
                    # of the word, so 'CAR' does not match inside 'F008_update'
                    # or 'my-CAR-thing'.
                    pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(term.strip()) + r'(?![a-zA-Z0-9_-])'
                    try:
                        regex = re.compile(pattern, re.IGNORECASE)
                        compiled_terms.append((term.strip(), regex))
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern for term '{term}': {e}")
            
            compiled[store] = compiled_terms
        
        return compiled
    
    def extract(self, content, metadata=None):
        """Extract domain terms from content based on store in metadata."""
        md = metadata or {}
        store = md.get('store')
        
        # If store is specified (and not empty), use only that store with default fallback
        if store:
            rules = self._compiled.get(store, self._compiled.get('default', []))
        else:
            # If no store specified, use union of all stores except 'default'
            # (default is empty by design and should not be included in union)
            rules = []
            seen_terms = set()
            for store_name, store_rules in self._compiled.items():
                if store_name != 'default':  # Skip empty default store
                    for term, regex in store_rules:
                        if term not in seen_terms:  # Dedup by name
                            rules.append((term, regex))
                            seen_terms.add(term)
        
        entities = []
        for term, regex in rules:
            if regex.search(content):
                entities.append(Entity(
                    name=term,
                    entity_type='domain_term',
                    source='domain'
                ))
        
        return entities