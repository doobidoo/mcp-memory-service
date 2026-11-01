-- Migration 020: Add embedding column to beliefs (semantic relevance in belief re-rank)
-- Gives each belief a stored embedding vector (serialize_float32 BLOB) so a later re-rank
-- can fuse lexical overlap with semantic cosine similarity between query and belief.
-- NULL means not yet backfilled and the re-rank falls back to lexical-only for that row.
-- Additive and reversible, no existing column or constraint is touched.

ALTER TABLE beliefs ADD COLUMN embedding BLOB DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_beliefs_embedding_pending
    ON beliefs(belief_hash) WHERE embedding IS NULL;
