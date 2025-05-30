"""
Database models for EchoVault Memory Service.
Copyright (c) 2025 EchoVault
Licensed under the MIT License.
"""

from sqlalchemy import Column, String, Text, Integer, BigInteger, TIMESTAMP, text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

# Create a base class for declarative models
Base = declarative_base()

class Memory(Base):
    """Memory model for storing memory events."""
    __tablename__ = 'memories'
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(float), nullable=True)  # Will be converted to vector(1536) in migration
    metadata = Column(JSONB, nullable=True)
    tags = Column(JSONB, nullable=True)
    timestamp = Column(BigInteger, nullable=True)
    content_hash = Column(String, unique=True, nullable=False)
    memory_type = Column(String, nullable=True)
    payload_url = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)
    
    # Indexes defined in migrations

class Telemetry(Base):
    """Telemetry model for storing event data."""
    __tablename__ = 'telemetry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    event_data = Column(JSONB, nullable=True)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)

class MemorySummary(Base):
    """Memory summary model for storing summarized memories."""
    __tablename__ = 'memory_summaries'
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(float), nullable=True)  # Will be converted to vector(1536) in migration
    metadata = Column(JSONB, nullable=True)
    tags = Column(JSONB, nullable=True)
    source_memories = Column(JSONB, nullable=True)  # Store IDs of original memories
    start_timestamp = Column(BigInteger, nullable=True)
    end_timestamp = Column(BigInteger, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)