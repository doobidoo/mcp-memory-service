# Hybrid Memory Consolidation with On-Device SLMs

## Overview

This document extends the [Autonomous Memory Consolidation](./autonomous-memory-consolidation.md) system by selectively incorporating on-device Small Language Models (SLMs) to enhance natural language capabilities while maintaining privacy, efficiency, and autonomous operation.

> **Note**: This is an optional enhancement. The memory consolidation system works fully autonomously without SLMs, but can provide richer insights when enhanced with local AI models.

## Why Hybrid?

The autonomous system excels at:
- ✅ Mathematical operations (similarity, clustering)
- ✅ Deterministic behavior
- ✅ Zero dependencies
- ❌ Natural language summaries
- ❌ Complex reasoning about connections

On-device SLMs add:
- ✅ Eloquent prose summaries
- ✅ Nuanced understanding
- ✅ Creative insights
- ✅ Still completely private (local processing)

## Recommended On-Device SLMs

### Tier 1: Ultra-Lightweight (< 2GB RAM)

#### **Llama 3.2 1B-Instruct**
- **Size**: ~1.2GB quantized (Q4_K_M)
- **Performance**: 50-100 tokens/sec on CPU
- **Best for**: Basic summarization, keyword expansion
- **Install**: `ollama pull llama3.2:1b`

```python
import ollama

def generate_summary_with_llama(cluster_data):
    """Generate natural language summary for memory cluster."""
    prompt = f"""Summarize these key themes from related memories:
    Keywords: {', '.join(cluster_data['keywords'])}
    Time span: {cluster_data['time_span']}
    Number of memories: {cluster_data['count']}
    
    Provide a concise, insightful summary:"""
    
    response = ollama.generate(model='llama3.2:1b', prompt=prompt)
    return response['response']
```

#### **Phi-3-mini (3.8B)**
- **Size**: ~2.3GB quantized
- **Strengths**: Exceptional reasoning for size
- **Best for**: Analyzing creative connections
- **Install**: `ollama pull phi3:mini`

### Tier 2: Balanced Performance (4-8GB RAM)

#### **Mistral 7B-Instruct v0.3**
- **Size**: ~4GB quantized (Q4_K_M)
- **Performance**: 20-40 tokens/sec on modern CPU
- **Best for**: Full consolidation narratives
- **Install**: `ollama pull mistral:7b-instruct-q4_K_M`

```python
class MistralEnhancedConsolidator:
    def __init__(self):
        self.model = "mistral:7b-instruct-q4_K_M"
    
    async def create_consolidation_narrative(self, clusters, associations):
        """Create a narrative summary of the consolidation results."""
        prompt = f"""Based on memory consolidation analysis:
        
        Found {len(clusters)} memory clusters and {len(associations)} creative connections.
        
        Key themes: {self.extract_themes(clusters)}
        Surprising connections: {self.format_associations(associations[:3])}
        
        Write a brief narrative summary highlighting the most important insights 
        and patterns discovered during this consolidation cycle."""
        
        response = await ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.7, "max_tokens": 200}
        )
        return response['response']
```

#### **Gemma 2B**
- **Size**: ~1.5GB quantized
- **Strengths**: Google's training quality
- **Best for**: Classification and scoring
- **Install**: `ollama pull gemma:2b`

### Tier 3: High-Performance (8-16GB RAM)

#### **Qwen 2.5 7B-Instruct**
- **Size**: ~4-5GB quantized
- **Strengths**: Multilingual, complex reasoning
- **Best for**: International users, detailed analysis
- **Install**: `ollama pull qwen2.5:7b-instruct`

## Hybrid Implementation Architecture

```python
from enum import Enum
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

class ProcessingMode(Enum):
    AUTONOMOUS_ONLY = "autonomous"
    HYBRID_SELECTIVE = "hybrid_selective"
    HYBRID_FULL = "hybrid_full"

class HybridMemoryConsolidator:
    """
    Combines autonomous processing with selective SLM enhancement.
    
    The system always runs autonomous processing first, then selectively
    enhances results with SLM-generated insights where valuable.
    """
    
    def __init__(self, storage, config):
        # Core autonomous system (always available)
        self.autonomous = AutonomousMemoryConsolidator(storage, config)
        
        # SLM configuration (optional enhancement)
        self.mode = ProcessingMode(config.get('processing_mode', 'autonomous'))
        self.slm_model = config.get('slm_model', 'llama3.2:1b')
        self.slm_available = self._check_slm_availability()
        
        # Enhancement thresholds
        self.min_cluster_size = config.get('slm_min_cluster_size', 5)
        self.min_importance = config.get('slm_min_importance', 0.7)
        self.enhancement_horizons = config.get(
            'slm_time_horizons', 
            ['weekly', 'monthly', 'quarterly', 'yearly']
        )
    
    def _check_slm_availability(self) -> bool:
        """Check if SLM is available for enhancement."""
        if self.mode == ProcessingMode.AUTONOMOUS_ONLY:
            return False
            
        try:
            import ollama
            # Check if model is available
            models = ollama.list()
            return any(m['name'].startswith(self.slm_model) for m in models['models'])
        except:
            return False
    
    async def consolidate(self, time_horizon: str) -> Dict:
        """
        Run consolidation with optional SLM enhancement.
        
        Always performs autonomous processing first, then selectively
        enhances based on configuration and context.
        """
        # Step 1: Always run autonomous processing
        auto_results = await self.autonomous.consolidate(time_horizon)
        
        # Step 2: Determine if SLM enhancement should be applied
        if not self._should_enhance(time_horizon, auto_results):
            return auto_results
        
        # Step 3: Selective SLM enhancement
        enhanced_results = await self._enhance_with_slm(
            auto_results, 
            time_horizon
        )
        
        return enhanced_results
    
    def _should_enhance(self, time_horizon: str, results: Dict) -> bool:
        """Determine if SLM enhancement would add value."""
        # Check if SLM is available
        if not self.slm_available:
            return False
        
        # Check if time horizon warrants enhancement
        if time_horizon not in self.enhancement_horizons:
            return False
        
        # Check if results are significant enough
        significant_clusters = sum(
            1 for cluster in results.get('clusters', [])
            if len(cluster) >= self.min_cluster_size
        )
        
        return significant_clusters > 0
    
    async def _enhance_with_slm(self, auto_results: Dict, time_horizon: str) -> Dict:
        """Selectively enhance autonomous results with SLM insights."""
        enhanced = auto_results.copy()
        
        # Enhance cluster summaries
        if 'clusters' in enhanced:
            enhanced['narrative_summaries'] = []
            for i, cluster in enumerate(enhanced['clusters']):
                if len(cluster) >= self.min_cluster_size:
                    narrative = await self._generate_cluster_narrative(
                        cluster, 
                        auto_results.get('compressed_summaries', [])[i]
                    )
                    enhanced['narrative_summaries'].append({
                        'cluster_id': i,
                        'narrative': narrative,
                        'memory_count': len(cluster)
                    })
        
        # Enhance creative associations
        if 'associations' in enhanced and len(enhanced['associations']) > 0:
            insights = await self._generate_association_insights(
                enhanced['associations'][:5]  # Top 5 associations
            )
            enhanced['association_insights'] = insights
        
        # Generate consolidation overview
        enhanced['consolidation_narrative'] = await self._generate_overview(
            enhanced, 
            time_horizon
        )
        
        enhanced['processing_mode'] = 'hybrid'
        enhanced['slm_model'] = self.slm_model
        
        return enhanced
    
    async def _generate_cluster_narrative(
        self, 
        cluster: List, 
        compressed_summary: Dict
    ) -> str:
        """Generate natural language narrative for a memory cluster."""
        prompt = f"""Based on this memory cluster analysis:
        
        Keywords: {', '.join(compressed_summary['keywords'][:10])}
        Time span: {compressed_summary['temporal_range']['start']} to {compressed_summary['temporal_range']['end']}
        Common tags: {', '.join(compressed_summary['common_tags'][:5])}
        Number of memories: {len(cluster)}
        
        Create a brief, insightful summary that captures the essence of these 
        related memories and any patterns or themes you notice:"""
        
        response = await self._call_slm(prompt, max_tokens=150)
        return response
    
    async def _generate_association_insights(
        self, 
        associations: List[Dict]
    ) -> List[Dict]:
        """Generate insights about creative associations discovered."""
        insights = []
        
        for assoc in associations:
            prompt = f"""Two memories were found to have an interesting connection 
            (similarity: {assoc['similarity']:.2f}).
            
            Memory 1: {assoc['memory_1_preview'][:100]}...
            Memory 2: {assoc['memory_2_preview'][:100]}...
            
            What insight or pattern might this connection reveal?
            Be concise and focus on the non-obvious relationship:"""
            
            insight = await self._call_slm(prompt, max_tokens=80)
            insights.append({
                'association_id': assoc['id'],
                'insight': insight,
                'similarity': assoc['similarity']
            })
        
        return insights
    
    async def _generate_overview(
        self, 
        results: Dict, 
        time_horizon: str
    ) -> str:
        """Generate a narrative overview of the consolidation cycle."""
        prompt = f"""Memory consolidation {time_horizon} summary:
        
        - Processed {results.get('total_memories', 0)} memories
        - Found {len(results.get('clusters', []))} memory clusters
        - Discovered {len(results.get('associations', []))} creative connections
        - Archived {results.get('archived_count', 0)} low-relevance memories
        
        Key themes: {self._extract_top_themes(results)}
        
        Write a brief executive summary of this consolidation cycle, 
        highlighting the most important patterns and any surprising discoveries:"""
        
        response = await self._call_slm(prompt, max_tokens=200)
        return response
    
    async def _call_slm(self, prompt: str, max_tokens: int = 100) -> str:
        """Call the SLM with error handling."""
        try:
            import ollama
            response = ollama.generate(
                model=self.slm_model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "stop": ["\n\n", "###"]
                }
            )
            return response['response'].strip()
        except Exception as e:
            # Fallback to autonomous summary
            return f"[SLM unavailable: {str(e)}]"
    
    def _extract_top_themes(self, results: Dict) -> str:
        """Extract top themes from results."""
        all_keywords = []
        for summary in results.get('compressed_summaries', []):
            all_keywords.extend(summary.get('keywords', []))
        
        # Count frequency
        from collections import Counter
        theme_counts = Counter(all_keywords)
        top_themes = [theme for theme, _ in theme_counts.most_common(5)]
        
        return ', '.join(top_themes) if top_themes else 'various topics'
```

## Smart Enhancement Strategy

```python
class SmartEnhancementStrategy:
    """
    Intelligently decide when and how to use SLM enhancement.
    
    Principles:
    1. Autonomous processing is always the foundation
    2. SLM enhancement only when it adds significant value
    3. Resource usage scales with importance
    """
    
    def __init__(self, config):
        self.config = config
        
        # Enhancement criteria
        self.criteria = {
            'min_cluster_size': 5,
            'min_importance_score': 0.7,
            'min_association_similarity': 0.4,
            'max_association_similarity': 0.7,
            'enhancement_time_horizons': ['weekly', 'monthly', 'quarterly', 'yearly'],
            'daily_enhancement': False,  # Too frequent
            'require_user_request': False
        }
    
    def should_enhance_cluster(self, cluster_info: Dict) -> bool:
        """Decide if a cluster warrants SLM enhancement."""
        # Size check
        if cluster_info['size'] < self.criteria['min_cluster_size']:
            return False
        
        # Importance check
        avg_importance = np.mean([m.importance_score for m in cluster_info['memories']])
        if avg_importance < self.criteria['min_importance_score']:
            return False
        
        # Complexity check (high variance suggests interesting cluster)
        embedding_variance = np.var([m.embedding for m in cluster_info['memories']], axis=0).mean()
        if embedding_variance < 0.1:  # Too homogeneous
            return False
        
        return True
    
    def select_model_for_task(self, task_type: str, resource_limit: str) -> str:
        """Select appropriate model based on task and resources."""
        model_selection = {
            'basic_summary': {
                'low': 'llama3.2:1b',
                'medium': 'phi3:mini',
                'high': 'mistral:7b-instruct'
            },
            'creative_insights': {
                'low': 'phi3:mini',  # Good reasoning even when small
                'medium': 'mistral:7b-instruct',
                'high': 'qwen2.5:7b-instruct'
            },
            'technical_analysis': {
                'low': 'gemma:2b',
                'medium': 'mistral:7b-instruct',
                'high': 'qwen2.5:7b-instruct'
            }
        }
        
        return model_selection.get(task_type, {}).get(resource_limit, 'llama3.2:1b')
```

## Configuration Examples

### Minimal Enhancement (Low Resources)
```yaml
hybrid_consolidation:
  processing_mode: "hybrid_selective"
  slm_model: "llama3.2:1b"
  slm_min_cluster_size: 10  # Only largest clusters
  slm_min_importance: 0.8   # Only most important
  slm_time_horizons: ["monthly", "quarterly"]  # Less frequent
  max_tokens_per_summary: 100
```

### Balanced Enhancement (Recommended)
```yaml
hybrid_consolidation:
  processing_mode: "hybrid_selective"
  slm_model: "mistral:7b-instruct-q4_K_M"
  slm_min_cluster_size: 5
  slm_min_importance: 0.7
  slm_time_horizons: ["weekly", "monthly", "quarterly", "yearly"]
  max_tokens_per_summary: 150
  enable_creative_insights: true
  enable_narrative_summaries: true
```

### Full Enhancement (High Resources)
```yaml
hybrid_consolidation:
  processing_mode: "hybrid_full"
  slm_model: "qwen2.5:7b-instruct"
  slm_min_cluster_size: 3
  slm_min_importance: 0.5
  slm_time_horizons: ["daily", "weekly", "monthly", "quarterly", "yearly"]
  max_tokens_per_summary: 200
  enable_creative_insights: true
  enable_narrative_summaries: true
  enable_predictive_insights: true
  parallel_processing: true
```

## Installation Guide

### Using Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models based on your resources
# Minimal (2GB)
ollama pull llama3.2:1b

# Balanced (8GB)
ollama pull mistral:7b-instruct-q4_K_M

# High-performance (16GB)
ollama pull qwen2.5:7b-instruct

# Test the model
ollama run llama3.2:1b "Summarize: AI helps organize memories"
```

### Using llama.cpp
```python
from llama_cpp import Llama

# Initialize with specific model
llm = Llama(
    model_path="./models/llama-3.2-1b-instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=-1  # Use GPU if available
)

# Generate summary
response = llm(
    prompt="Summarize these themes: productivity, learning, coding",
    max_tokens=100,
    temperature=0.7
)
```

## Performance Considerations

### Resource Usage by Model

| Model | RAM Usage | CPU Tokens/sec | GPU Tokens/sec | Quality |
|-------|-----------|----------------|----------------|---------|
| Llama 3.2 1B | 1.2GB | 50-100 | 200-400 | Good |
| Phi-3 mini | 2.3GB | 30-60 | 150-300 | Excellent |
| Mistral 7B Q4 | 4GB | 20-40 | 100-200 | Excellent |
| Gemma 2B | 1.5GB | 40-80 | 180-350 | Good |
| Qwen 2.5 7B | 5GB | 15-30 | 80-150 | Best |

### Optimization Strategies

1. **Batch Processing**: Process multiple summaries in one call
2. **Caching**: Cache SLM responses for similar inputs
3. **Progressive Enhancement**: Start with fast model, upgrade if needed
4. **Time-based Scheduling**: Run SLM enhancement during off-hours

## Benefits of Hybrid Approach

### ✅ **Advantages**
1. **Best of Both Worlds**: Mathematical precision + natural language eloquence
2. **Flexible Deployment**: Can disable SLM without breaking system
3. **Privacy Preserved**: Everything runs locally
4. **Resource Efficient**: SLM only when valuable
5. **Progressive Enhancement**: Better with SLM, functional without

### 📊 **Comparison**

| Feature | Autonomous Only | Hybrid with SLM |
|---------|----------------|-----------------|
| Natural summaries | ❌ Structured data | ✅ Eloquent prose |
| Creative insights | ❌ Statistical only | ✅ Nuanced understanding |
| Resource usage | ✅ Minimal | 🔶 Moderate |
| Speed | ✅ Very fast | 🔶 Task-dependent |
| Deterministic | ✅ Always | 🔶 Core operations only |
| Privacy | ✅ Complete | ✅ Complete |

## Example Output Comparison

### Autonomous Only
```json
{
  "cluster_summary": {
    "keywords": ["python", "debugging", "memory", "optimization"],
    "memory_count": 8,
    "time_span": "2025-07-21 to 2025-07-28",
    "representative_memory": "Fixed memory leak in consolidation engine"
  }
}
```

### Hybrid with SLM
```json
{
  "cluster_summary": {
    "keywords": ["python", "debugging", "memory", "optimization"],
    "memory_count": 8,
    "time_span": "2025-07-21 to 2025-07-28",
    "representative_memory": "Fixed memory leak in consolidation engine",
    "narrative": "This week focused on resolving critical performance issues in the memory consolidation system. The memory leak in the clustering algorithm was traced to improper cleanup of embedding vectors, resulting in a 40% performance improvement after the fix. These debugging sessions revealed important patterns about resource management in long-running consolidation processes.",
    "key_insight": "Proper lifecycle management of vector embeddings is crucial for maintaining performance in continuous consolidation systems."
  }
}
```

## Future Enhancements

1. **Fine-tuned Models**: Train small models specifically for memory consolidation
2. **Multi-Model Ensemble**: Use different models for different tasks
3. **Adaptive Model Selection**: Automatically choose model based on task complexity
4. **Streaming Generation**: Process summaries as they generate
5. **Quantization Optimization**: Test various quantization levels for best trade-offs

## Conclusion

The hybrid approach with on-device SLMs provides the perfect balance between the reliability of autonomous processing and the expressiveness of natural language AI. By running everything locally and using SLMs selectively, we maintain privacy, control costs, and ensure the system remains functional even without AI enhancement.

This transforms the dream-inspired memory consolidation from a purely algorithmic system into an intelligent assistant that can provide genuine insights while respecting user privacy and system resources.

---

**Related Documents:**
- 🔧 [Autonomous Memory Consolidation Guide](./autonomous-memory-consolidation.md)
- 💭 [Dream-Inspired Memory Consolidation System](./dream-inspired-memory-consolidation.md)
- 📋 [Issue #11: Multi-Layered Memory Consolidation](https://github.com/doobidoo/mcp-memory-service/issues/11)

*Created: July 28, 2025*