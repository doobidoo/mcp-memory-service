# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compact CSV encoding for quality and consolidation metadata.

Reduces metadata size by 60% compared to JSON while maintaining readability.
Used for Cloudflare sync to stay under 10KB D1 metadata limit.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Provider code mapping (70% size reduction)
PROVIDER_CODES = {
    'onnx_local': 'ox',
    'implicit_signals': 'im',
    'groq': 'gq',
    'gemini': 'gm',
    'local': 'lc',
    'auto': 'au',
    'none': 'no',
    'fallback_deberta-msmarco': 'fb',  # Fallback mode (DeBERTa + MS-MARCO)
    'onnx_deberta': 'od',              # DeBERTa-only scoring
    'onnx_msmarco': 'om'               # Legacy MS-MARCO-only scoring
}

PROVIDER_DECODE = {v: k for k, v in PROVIDER_CODES.items()}

# Decision code mapping for fallback scoring
DECISION_CODES = {
    'deberta_confident': 'dc',
    'ms_marco_rescue': 'mr',
    'both_low': 'bl',
    'deberta_only': 'do',
    'ms_marco_failed': 'mf'
}

DECISION_DECODE = {v: k for k, v in DECISION_CODES.items()}


def encode_quality_metadata(metadata: Dict[str, Any]) -> str:
    """
    Encode quality and consolidation metadata to compact CSV format.

    Format: qs,qp,as_scores,rs,rca,df,cb,ab,qba,qbd,qbr,qbcc,oqbb,dec,dbs,mms

    Fields:
        qs   - quality_score
        qp   - quality_provider (encoded)
        as   - ai_scores (last 3)
        rs   - relevance_score
        rca  - relevance_calculated_at
        df   - decay_factor
        cb   - connection_boost
        ab   - access_boost
        qba  - quality_boost_applied
        qbd  - quality_boost_date
        qbr  - quality_boost_reason
        qbcc - quality_boost_connection_count
        oqbb - original_quality_before_boost
        dec  - decision (fallback mode only)
        dbs  - deberta_score (fallback mode only)
        mms  - ms_marco_score (fallback mode only)

    Example:
        0.851,ox,0.997:1733583071;0.995:1733583072,0.87,1733583071,0.95,0.12,0.05,1,1733583071,assoc,5,0.750,,,
        0.723,fb,,,,,,,,,,,mr,0.58,0.85

    Args:
        metadata: Full metadata dict with quality/consolidation fields

    Returns:
        Compact CSV string (~110-130 bytes vs ~180-220 bytes JSON)
    """
    parts = []

    # Quality score (required)
    parts.append(str(metadata.get('quality_score', 0.5)))

    # Quality provider (encode to short code)
    provider = metadata.get('quality_provider', 'im')
    parts.append(PROVIDER_CODES.get(provider, 'im'))

    # AI scores history (keep last 3 instead of 10)
    ai_scores = metadata.get('ai_scores', [])
    if ai_scores:
        # Format: score:timestamp;score:timestamp;...
        ai_parts = []
        for score_entry in ai_scores[-3:]:  # Only last 3
            s = score_entry.get('score', 0)
            t = int(score_entry.get('timestamp', 0))
            ai_parts.append(f"{s}:{t}")
        parts.append(';'.join(ai_parts))
    else:
        parts.append('')

    # Relevance score (consolidation)
    parts.append(str(metadata.get('relevance_score', '')))

    # Relevance calculated at (convert ISO to Unix timestamp if present)
    rca = metadata.get('relevance_calculated_at', '')
    if rca:
        # If it's ISO format, convert to Unix timestamp
        try:
            from datetime import datetime
            if 'T' in str(rca):
                dt = datetime.fromisoformat(rca.replace('Z', '+00:00'))
                rca = int(dt.timestamp())
        except (ValueError, TypeError, AttributeError):
            pass  # Keep original rca value if datetime parsing fails
    parts.append(str(rca))

    # Decay factors
    parts.append(str(metadata.get('decay_factor', '')))
    parts.append(str(metadata.get('connection_boost', '')))
    parts.append(str(metadata.get('access_boost', '')))

    # Quality boost fields (association boost)
    parts.append('1' if metadata.get('quality_boost_applied') else '')

    qbd = metadata.get('quality_boost_date', '')
    if qbd:
        try:
            from datetime import datetime
            if 'T' in str(qbd):
                dt = datetime.fromisoformat(qbd.replace('Z', '+00:00'))
                qbd = int(dt.timestamp())
        except (ValueError, TypeError, AttributeError):
            pass  # Keep original qbd value if datetime parsing fails
    parts.append(str(qbd))

    parts.append(str(metadata.get('quality_boost_reason', '')))
    parts.append(str(metadata.get('quality_boost_connection_count', '')))
    parts.append(str(metadata.get('original_quality_before_boost', '')))

    # Fallback-specific fields (quality_components)
    components = metadata.get('quality_components', {})
    if components:
        # Decision code (dc, mr, bl, do, mf)
        decision = components.get('decision', '')
        parts.append(DECISION_CODES.get(decision, ''))

        # Individual model scores (for analysis/debugging)
        deberta_score = components.get('deberta_score')
        parts.append(f"{deberta_score:.3f}" if deberta_score is not None else '')

        ms_marco_score = components.get('ms_marco_score')
        parts.append(f"{ms_marco_score:.3f}" if ms_marco_score is not None else '')
    else:
        # No fallback data - append empty fields
        parts.append('')  # decision
        parts.append('')  # deberta_score
        parts.append('')  # ms_marco_score

    return ','.join(parts)


def decode_quality_metadata(csv_string: str) -> Dict[str, Any]:
    """
    Decode compact CSV format back to metadata dict.

    Args:
        csv_string: Compact CSV string from encode_quality_metadata()

    Returns:
        Metadata dict with all quality/consolidation fields
    """
    if not csv_string:
        return {}

    parts = csv_string.split(',')
    # Support both old format (13 parts) and new format (16 parts)
    if len(parts) < 13:
        logger.warning(f"Invalid CSV metadata format: expected at least 13 parts, got {len(parts)}")
        return {}

    metadata = {}

    try:
        # Quality score
        if parts[0]:
            metadata['quality_score'] = float(parts[0])

        # Quality provider (decode from short code)
        if parts[1]:
            metadata['quality_provider'] = PROVIDER_DECODE.get(parts[1], 'implicit_signals')

        # AI scores history
        if parts[2]:
            ai_scores = []
            for score_part in parts[2].split(';'):
                if ':' in score_part:
                    s, t = score_part.split(':', 1)
                    ai_scores.append({
                        'score': float(s),
                        'timestamp': int(t),
                        'provider': metadata.get('quality_provider', 'implicit_signals')
                    })
            metadata['ai_scores'] = ai_scores

        # Relevance score
        if parts[3]:
            metadata['relevance_score'] = float(parts[3])

        # Relevance calculated at (Unix timestamp -> ISO)
        if parts[4]:
            from datetime import datetime
            timestamp = int(parts[4])
            metadata['relevance_calculated_at'] = datetime.fromtimestamp(timestamp).isoformat()

        # Decay factors
        if parts[5]:
            metadata['decay_factor'] = float(parts[5])
        if parts[6]:
            metadata['connection_boost'] = float(parts[6])
        if parts[7]:
            metadata['access_boost'] = float(parts[7])

        # Quality boost fields
        if parts[8] == '1':
            metadata['quality_boost_applied'] = True

        if parts[9]:
            from datetime import datetime
            timestamp = int(parts[9])
            metadata['quality_boost_date'] = datetime.fromtimestamp(timestamp).isoformat()

        if parts[10]:
            metadata['quality_boost_reason'] = parts[10]
        if parts[11]:
            metadata['quality_boost_connection_count'] = int(parts[11])
        if parts[12]:
            metadata['original_quality_before_boost'] = float(parts[12])

        # Fallback-specific fields (new format only - 16 parts)
        if len(parts) >= 16:
            components = {}

            # Decision code
            if parts[13]:
                components['decision'] = DECISION_DECODE.get(parts[13], parts[13])

            # DeBERTa score
            if parts[14]:
                components['deberta_score'] = float(parts[14])

            # MS-MARCO score
            if parts[15]:
                components['ms_marco_score'] = float(parts[15])

            # Only add quality_components if we have any fallback data
            if components:
                # Add final_score for consistency
                components['final_score'] = metadata.get('quality_score')
                metadata['quality_components'] = components

    except (ValueError, IndexError) as e:
        logger.error(f"Error decoding CSV metadata: {e}")
        return {}

    return metadata


def compress_metadata_for_sync(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compress quality/consolidation metadata for Cloudflare sync.

    Replaces verbose JSON fields with compact CSV in '_q' field.
    Removes debug-only fields (quality_components).

    Args:
        metadata: Full metadata dict

    Returns:
        Compressed metadata dict (60% smaller)
    """
    # Create a copy
    compressed = metadata.copy()

    # Remove debug-only fields (not needed in Cloudflare)
    compressed.pop('quality_components', None)

    # Extract quality/consolidation fields
    quality_fields = [
        'quality_score', 'quality_provider', 'ai_scores',
        'relevance_score', 'relevance_calculated_at',
        'decay_factor', 'connection_boost', 'access_boost',
        'quality_boost_applied', 'quality_boost_date',
        'quality_boost_reason', 'quality_boost_connection_count',
        'original_quality_before_boost'
    ]

    # Check if any quality fields exist
    has_quality_data = any(field in metadata for field in quality_fields)

    if has_quality_data:
        # Encode to CSV
        csv_encoded = encode_quality_metadata(metadata)
        compressed['_q'] = csv_encoded

        # Remove original verbose fields
        for field in quality_fields:
            compressed.pop(field, None)

    return compressed


def decompress_metadata_from_sync(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompress quality metadata from Cloudflare sync.

    Expands compact '_q' CSV field back to full JSON fields.

    Args:
        metadata: Compressed metadata dict from Cloudflare

    Returns:
        Full metadata dict with all fields expanded
    """
    if not metadata:
        return {}

    # Create a copy
    decompressed = metadata.copy()

    # Check for compressed quality data
    if '_q' in decompressed:
        csv_data = decompressed.pop('_q')
        quality_fields = decode_quality_metadata(csv_data)
        decompressed.update(quality_fields)

    return decompressed
