"""
Model Tier Configuration
Contains metadata about recommended model tiers for future UI features

This file preserves the tier structure for potential future use.
Currently, the UI uses individual model installation via model_switcher.py
"""

from typing import Dict, List, Any
from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    size: str
    description: str
    use_cases: List[str]
    requirements: Dict[str, Any]
    recommended_hardware: str
    tool_support: bool = False
    license: str = "Apache 2.0"


class ModelTier(BaseModel):
    tier_name: str
    display_name: str
    description: str
    total_size: str
    disk_space_gb: int
    ram_requirements_gb: int
    models: List[ModelInfo]
    installation_time_estimate: str


# Model tier definitions - preserved for potential future tier-based UI
MODEL_TIERS = {
    "essential": ModelTier(
        tier_name="essential",
        display_name="🚀 Essential Pack",
        description="Lightweight models with full tool calling support for Roampal's memory system.",
        total_size="~6.7GB",
        disk_space_gb=10,
        ram_requirements_gb=4,
        installation_time_estimate="5-8 minutes",
        models=[
            ModelInfo(
                name="llama3.2:3b",
                size="2.0GB",
                description="✅ Tool Support - Compact model with excellent tool calling",
                use_cases=["Memory search", "Tool calling", "General chat", "Fast responses"],
                requirements={"ram_gb": 3, "vram_gb": 2},
                recommended_hardware="8GB+ RAM",
                tool_support=True,
                license="Meta Custom License"
            ),
            ModelInfo(
                name="qwen2.5:3b",
                size="1.9GB",
                description="✅ Tool Support - Alibaba's efficient model with native tools",
                use_cases=["Tool calling", "Memory operations", "Multi-language", "Reasoning"],
                requirements={"ram_gb": 3, "vram_gb": 2},
                recommended_hardware="8GB+ RAM",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="nomic-embed-text:latest",
                size="274MB",
                description="Essential embedding model for Roampal's vector memory",
                use_cases=["Memory indexing", "Semantic search", "Document embeddings"],
                requirements={"ram_gb": 1, "vram_gb": 0.3},
                recommended_hardware="Any modern computer",
                tool_support=False,
                license="Apache 2.0"
            )
        ]
    ),

    "professional": ModelTier(
        tier_name="professional",
        display_name="⚡ Professional Pack",
        description="Balanced performance models with excellent tool calling for Roampal's memory system.",
        total_size="~35GB",
        disk_space_gb=45,
        ram_requirements_gb=8,
        installation_time_estimate="15-20 minutes",
        models=[
            ModelInfo(
                name="gpt-oss:20b",
                size="11GB",
                description="✅ Tool Support - OpenAI's first open source model with native tools",
                use_cases=["Tool calling", "Function calling", "Memory search", "Reasoning"],
                requirements={"ram_gb": 16, "vram_gb": 8},
                recommended_hardware="16GB+ RAM",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="qwen2.5:7b",
                size="4.4GB",
                description="✅ Tool Support - Best-in-class tool calling performance",
                use_cases=["Tool calling", "Complex reasoning", "Multi-language", "Memory ops"],
                requirements={"ram_gb": 6, "vram_gb": 3},
                recommended_hardware="12GB+ RAM",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="llama3.1:8b",
                size="4.9GB",
                description="✅ Tool Support - Meta's balanced model with native tools",
                use_cases=["Tool calling", "General purpose", "Code assistance", "Memory search"],
                requirements={"ram_gb": 6, "vram_gb": 3},
                recommended_hardware="12GB+ RAM",
                tool_support=True,
                license="Meta Custom License"
            ),
            ModelInfo(
                name="qwen2.5:14b",
                size="8.9GB",
                description="✅ Tool Support - Larger Qwen with superior tool performance",
                use_cases=["Advanced tool calling", "Complex tasks", "Professional work"],
                requirements={"ram_gb": 12, "vram_gb": 6},
                recommended_hardware="16GB+ RAM",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="phi4:14b",
                size="7.9GB",
                description="✅ Tool Support - Microsoft's Phi-4 with function calling",
                use_cases=["Tool calling", "Math reasoning", "Code generation", "Analysis"],
                requirements={"ram_gb": 10, "vram_gb": 5},
                recommended_hardware="16GB+ RAM",
                tool_support=True,
                license="MIT"
            )
        ]
    ),

    "enterprise": ModelTier(
        tier_name="enterprise",
        display_name="🔥 Enterprise Pack",
        description="Maximum capability models with advanced tool calling for demanding tasks.",
        total_size="~150GB",
        disk_space_gb=200,
        ram_requirements_gb=32,
        installation_time_estimate="45-60 minutes",
        models=[
            ModelInfo(
                name="gpt-oss:120b",
                size="80GB",
                description="✅ Tool Support - OpenAI's flagship open source model",
                use_cases=["Advanced tool calling", "Complex reasoning", "Enterprise tasks"],
                requirements={"ram_gb": 80, "vram_gb": 40},
                recommended_hardware="80GB+ RAM or H100 GPU",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="llama3.1:70b",
                size="40GB",
                description="✅ Tool Support - Meta's large model with native tools",
                use_cases=["Tool calling", "Complex analysis", "Research", "Professional writing"],
                requirements={"ram_gb": 32, "vram_gb": 16},
                recommended_hardware="32GB+ RAM, RTX 4090",
                tool_support=True,
                license="Meta Custom License"
            ),
            ModelInfo(
                name="qwen2.5:32b",
                size="18GB",
                description="✅ Tool Support - Powerful Qwen with excellent tools",
                use_cases=["Advanced tool calling", "Multi-language", "Complex tasks"],
                requirements={"ram_gb": 24, "vram_gb": 12},
                recommended_hardware="32GB+ RAM",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="qwen2.5:72b",
                size="41GB",
                description="✅ Tool Support - Massive Qwen for enterprise use",
                use_cases=["Enterprise tool calling", "Research", "Complex reasoning"],
                requirements={"ram_gb": 48, "vram_gb": 24},
                recommended_hardware="64GB+ RAM",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="mixtral:8x7b",
                size="26GB",
                description="✅ Tool Support - MoE architecture with native tools",
                use_cases=["Tool calling", "Multi-domain expertise", "Complex tasks"],
                requirements={"ram_gb": 24, "vram_gb": 12},
                recommended_hardware="32GB+ RAM, RTX 4090",
                tool_support=True,
                license="Apache 2.0"
            ),
            ModelInfo(
                name="command-r:35b",
                size="20GB",
                description="✅ Tool Support - Cohere's model optimized for tool use",
                use_cases=["RAG systems", "Multi-step tool use", "Agent workflows"],
                requirements={"ram_gb": 24, "vram_gb": 12},
                recommended_hardware="32GB+ RAM",
                tool_support=True,
                license="CC-BY-NC-4.0"
            )
        ]
    )
}


# Specialist models - preserved for reference
SPECIALIST_MODELS = {
    "firefunction": ModelInfo(
        name="firefunction-v2",
        size="8.5GB",
        description="✅ Tool Support - Optimized for function calling, rivals GPT-4o",
        use_cases=["Function calling", "Tool use", "API interactions", "Structured output"],
        requirements={"ram_gb": 10, "vram_gb": 5},
        recommended_hardware="16GB+ RAM",
        tool_support=True,
        license="Apache 2.0"
    )
}
