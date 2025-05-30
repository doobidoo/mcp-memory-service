"""
Create EchoVault Schema Migration
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This migration creates the necessary database schema for the EchoVault Memory Service.
"""

from alembic import op
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy.sql import text

# Revision identifiers
revision = '20251201120000'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Upgrade to this revision."""
    # Enable pgvector extension if it doesn't exist
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create memories table
    op.create_table('memories',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', pg.ARRAY(sa.Float()), nullable=True),
        sa.Column('metadata', pg.JSONB(), nullable=True),
        sa.Column('tags', pg.JSONB(), nullable=True),
        sa.Column('timestamp', sa.BigInteger(), nullable=True),
        sa.Column('content_hash', sa.String(), unique=True, nullable=False),
        sa.Column('memory_type', sa.String(), nullable=True),
        sa.Column('payload_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    
    # Create indexes
    op.create_index('idx_memories_content_hash', 'memories', ['content_hash'])
    op.create_index('idx_memories_timestamp', 'memories', ['timestamp'])
    
    # Create GIN index on tags
    op.execute('CREATE INDEX idx_memories_tags ON memories USING GIN (tags)')
    
    # Create vector index
    op.execute("""
    ALTER TABLE memories 
    ALTER COLUMN embedding TYPE vector(1536) 
    USING embedding::vector(1536)
    """)
    
    op.execute("""
    CREATE INDEX idx_memories_embedding ON memories 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100)
    """)
    
    # Create telemetry table
    op.create_table('telemetry',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('event_data', pg.JSONB(), nullable=True),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    
    # Create index on event_type
    op.create_index('idx_telemetry_event_type', 'telemetry', ['event_type'])
    op.create_index('idx_telemetry_timestamp', 'telemetry', ['timestamp'])
    
    # Create memory_summaries table for storing summarized memories
    op.create_table('memory_summaries',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', pg.ARRAY(sa.Float()), nullable=True),
        sa.Column('metadata', pg.JSONB(), nullable=True),
        sa.Column('tags', pg.JSONB(), nullable=True),
        sa.Column('source_memories', pg.JSONB(), nullable=True),
        sa.Column('start_timestamp', sa.BigInteger(), nullable=True),
        sa.Column('end_timestamp', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    
    # Create indexes on summary table
    op.create_index('idx_memory_summaries_timestamps', 'memory_summaries', ['start_timestamp', 'end_timestamp'])
    
    # Set up the vector column for summaries
    op.execute("""
    ALTER TABLE memory_summaries 
    ALTER COLUMN embedding TYPE vector(1536) 
    USING embedding::vector(1536)
    """)
    
    op.execute("""
    CREATE INDEX idx_memory_summaries_embedding ON memory_summaries 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100)
    """)


def downgrade() -> None:
    """Downgrade to the previous revision."""
    # Drop tables
    op.drop_table('memory_summaries')
    op.drop_table('telemetry')
    op.drop_table('memories')