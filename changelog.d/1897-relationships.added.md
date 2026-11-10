- **Relationship types and inference type patterns can live in the memory database (#1897, phase 3).**
  `SqliteVecMemoryStorage.add_relationship_type()` / `remove_relationship_type()` store a
  relationship with its `valid_patterns` and symmetry as a `relationship` row in
  `ontology_entries`, and `set_type_pattern()` / `clear_type_pattern()` store the base-type
  pairs relationship inference maps to a relationship (`type_pattern` rows). Both merge after
  the built-ins and only add: built-in relationships, their patterns and the built-in pairs
  cannot be replaced. `validate_relationship`, `is_allowed_pair` and
  `is_symmetric_relationship` read the merged set, and the inline `type_patterns` dict moved
  to `TYPE_PATTERNS` in `models/ontology.py`. `has_entity` is now a proper entry in
  `RELATIONSHIPS`; type patterns cannot name it, since its target is an entity key.
