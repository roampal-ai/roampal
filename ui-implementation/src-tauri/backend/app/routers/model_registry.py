"""
Unified Model Registry API
Single source of truth for model metadata across all providers.

Features:
- Model catalog with quantization options
- VRAM/GPU detection and recommendations
- Smart model selection based on hardware
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import asyncio
import logging
import subprocess
import re
import sys

from config.model_contexts import (
    MODEL_CONTEXTS,
    get_context_size,
    get_cached_vision,
    get_cached_tools,
    fetch_ollama_capabilities,
    probe_vision,
    _save_caps_cache,
    _caps_cache_key,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model-registry"])

# Windows-specific: Hide terminal windows when spawning subprocesses
_SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    _SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW

# ============================================================================
# QUANTIZATION OPTIONS - All available quants per model family
# ============================================================================
# Format: {base_model: {quant_level: {size_gb, vram_required_gb, quality_rating}}}
# Quality ratings: 1-5 (5=best quality, 1=most compressed)

QUANTIZATION_OPTIONS = {
    # Format: "ollama_tag" is the tag to pull from Ollama (e.g., qwen2.5:7b-instruct-q4_K_M)
    # The base model name (e.g., qwen2.5:7b) is the default Q4_K_M quant
    "qwen2.5:7b": {
        "Q2_K": {"size_gb": 2.8, "vram_gb": 3.5, "quality": 1, "file": "Qwen2.5-7B-Instruct-Q2_K.gguf", "ollama_tag": "qwen2.5:7b-instruct-q2_K"},
        "Q3_K_M": {"size_gb": 3.4, "vram_gb": 4.0, "quality": 2, "file": "Qwen2.5-7B-Instruct-Q3_K_M.gguf", "ollama_tag": "qwen2.5:7b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 4.68, "vram_gb": 5.5, "quality": 3, "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen2.5:7b"},
        "Q5_K_M": {"size_gb": 5.3, "vram_gb": 6.0, "quality": 4, "file": "Qwen2.5-7B-Instruct-Q5_K_M.gguf", "ollama_tag": "qwen2.5:7b-instruct-q5_K_M"},
        "Q6_K": {"size_gb": 6.1, "vram_gb": 7.0, "quality": 4, "file": "Qwen2.5-7B-Instruct-Q6_K.gguf", "ollama_tag": "qwen2.5:7b-instruct-q6_K"},
        "Q8_0": {"size_gb": 8.1, "vram_gb": 9.0, "quality": 5, "file": "Qwen2.5-7B-Instruct-Q8_0.gguf", "ollama_tag": "qwen2.5:7b-instruct-q8_0"},
    },
    "qwen2.5:14b": {
        "Q2_K": {"size_gb": 5.5, "vram_gb": 6.5, "quality": 1, "file": "Qwen2.5-14B-Instruct-Q2_K.gguf", "ollama_tag": "qwen2.5:14b-instruct-q2_K"},
        "Q3_K_M": {"size_gb": 6.6, "vram_gb": 7.5, "quality": 2, "file": "Qwen2.5-14B-Instruct-Q3_K_M.gguf", "ollama_tag": "qwen2.5:14b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 8.99, "vram_gb": 10.0, "quality": 3, "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen2.5:14b"},
        "Q5_K_M": {"size_gb": 10.5, "vram_gb": 12.0, "quality": 4, "file": "Qwen2.5-14B-Instruct-Q5_K_M.gguf", "ollama_tag": "qwen2.5:14b-instruct-q5_K_M"},
        "Q6_K": {"size_gb": 12.1, "vram_gb": 14.0, "quality": 4, "file": "Qwen2.5-14B-Instruct-Q6_K.gguf", "ollama_tag": "qwen2.5:14b-instruct-q6_K"},
        "Q8_0": {"size_gb": 15.7, "vram_gb": 18.0, "quality": 5, "file": "Qwen2.5-14B-Instruct-Q8_0.gguf", "ollama_tag": "qwen2.5:14b-instruct-q8_0"},
    },
    "qwen2.5:32b": {
        "Q2_K": {"size_gb": 12.2, "vram_gb": 14.0, "quality": 1, "file": "Qwen2.5-32B-Instruct-Q2_K.gguf", "ollama_tag": "qwen2.5:32b-instruct-q2_K"},
        "Q3_K_M": {"size_gb": 14.8, "vram_gb": 17.0, "quality": 2, "file": "Qwen2.5-32B-Instruct-Q3_K_M.gguf", "ollama_tag": "qwen2.5:32b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 19.9, "vram_gb": 22.0, "quality": 3, "file": "Qwen2.5-32B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen2.5:32b"},
        "Q5_K_M": {"size_gb": 23.7, "vram_gb": 26.0, "quality": 4, "file": "Qwen2.5-32B-Instruct-Q5_K_M.gguf", "ollama_tag": "qwen2.5:32b-instruct-q5_K_M"},
        "Q6_K": {"size_gb": 27.3, "vram_gb": 30.0, "quality": 4, "file": "Qwen2.5-32B-Instruct-Q6_K.gguf", "ollama_tag": "qwen2.5:32b-instruct-q6_K"},
    },
    "qwen2.5:72b": {
        "Q2_K": {"size_gb": 26.9, "vram_gb": 30.0, "quality": 1, "file": "Qwen2.5-72B-Instruct-Q2_K.gguf", "ollama_tag": "qwen2.5:72b-instruct-q2_K"},
        "Q3_K_M": {"size_gb": 33.2, "vram_gb": 36.0, "quality": 2, "file": "Qwen2.5-72B-Instruct-Q3_K_M.gguf", "ollama_tag": "qwen2.5:72b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 47.4, "vram_gb": 50.0, "quality": 3, "file": "Qwen2.5-72B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen2.5:72b"},
    },
    "qwen2.5:3b": {
        "Q4_K_M": {"size_gb": 1.93, "vram_gb": 3.0, "quality": 3, "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen2.5:3b"},
        "Q5_K_M": {"size_gb": 2.3, "vram_gb": 3.5, "quality": 4, "file": "Qwen2.5-3B-Instruct-Q5_K_M.gguf", "ollama_tag": "qwen2.5:3b-instruct-q5_K_M"},
        "Q8_0": {"size_gb": 3.4, "vram_gb": 4.5, "quality": 5, "file": "Qwen2.5-3B-Instruct-Q8_0.gguf", "ollama_tag": "qwen2.5:3b-instruct-q8_0"},
    },
    "llama3.2:3b": {
        "Q4_K_M": {"size_gb": 2.0, "vram_gb": 3.0, "quality": 3, "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "llama3.2:3b"},
        "Q5_K_M": {"size_gb": 2.4, "vram_gb": 3.5, "quality": 4, "file": "Llama-3.2-3B-Instruct-Q5_K_M.gguf", "ollama_tag": "llama3.2:3b-instruct-q5_K_M"},
        "Q8_0": {"size_gb": 3.5, "vram_gb": 4.5, "quality": 5, "file": "Llama-3.2-3B-Instruct-Q8_0.gguf", "ollama_tag": "llama3.2:3b-instruct-q8_0"},
    },
    "llama3.1:8b": {
        "Q3_K_M": {"size_gb": 3.6, "vram_gb": 4.5, "quality": 2, "file": "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf", "ollama_tag": "llama3.1:8b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 4.9, "vram_gb": 6.0, "quality": 3, "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "llama3.1:8b"},
        "Q5_K_M": {"size_gb": 5.7, "vram_gb": 7.0, "quality": 4, "file": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf", "ollama_tag": "llama3.1:8b-instruct-q5_K_M"},
        "Q6_K": {"size_gb": 6.6, "vram_gb": 8.0, "quality": 4, "file": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", "ollama_tag": "llama3.1:8b-instruct-q6_K"},
        "Q8_0": {"size_gb": 8.5, "vram_gb": 10.0, "quality": 5, "file": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", "ollama_tag": "llama3.1:8b-instruct-q8_0"},
    },
    "llama3.3:70b": {
        "Q2_K": {"size_gb": 24.0, "vram_gb": 28.0, "quality": 1, "file": "Llama-3.3-70B-Instruct-Q2_K.gguf", "ollama_tag": "llama3.3:70b-instruct-q2_K"},
        "Q3_K_M": {"size_gb": 30.5, "vram_gb": 34.0, "quality": 2, "file": "Llama-3.3-70B-Instruct-Q3_K_M.gguf", "ollama_tag": "llama3.3:70b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 42.5, "vram_gb": 46.0, "quality": 3, "file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "llama3.3:70b"},
    },
    "mixtral:8x7b": {
        "Q3_K_M": {"size_gb": 19.0, "vram_gb": 22.0, "quality": 2, "file": "mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf", "ollama_tag": "mixtral:8x7b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 26.44, "vram_gb": 30.0, "quality": 3, "file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", "default": True, "ollama_tag": "mixtral:8x7b"},
        "Q5_K_M": {"size_gb": 31.5, "vram_gb": 35.0, "quality": 4, "file": "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf", "ollama_tag": "mixtral:8x7b-instruct-q5_K_M"},
    },
    # OpenAI GPT-OSS - Uses native MXFP4 format (4.25 bits), no traditional quants
    # Both sizes available, 20b fits 16GB VRAM, 120b needs 80GB
    "gpt-oss:20b": {
        "MXFP4": {"size_gb": 14.0, "vram_gb": 16.0, "quality": 5, "file": "gpt-oss-20b", "default": True, "ollama_tag": "gpt-oss:20b"},
    },
    "gpt-oss:120b": {
        "MXFP4": {"size_gb": 65.0, "vram_gb": 80.0, "quality": 5, "file": "gpt-oss-120b", "default": True, "ollama_tag": "gpt-oss:120b"},
    },
    # Llama 4 Scout - MoE 109B total, 17B active, 10M context window
    "llama4:scout": {
        "UD-1.78bit": {"size_gb": 34.0, "vram_gb": 24.0, "quality": 2, "file": "llama4-scout-ud-1.78bit", "ollama_tag": "llama4:scout-ud"},
        "Q3_K_M": {"size_gb": 47.0, "vram_gb": 50.0, "quality": 2, "file": "Llama-4-Scout-17B-16E-Instruct-Q3_K_M.gguf", "ollama_tag": "llama4:scout-q3_K_M"},
        "Q4_K_M": {"size_gb": 65.0, "vram_gb": 70.0, "quality": 3, "file": "Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "llama4:scout"},
        "Q6_K": {"size_gb": 82.0, "vram_gb": 85.0, "quality": 4, "file": "Llama-4-Scout-17B-16E-Instruct-Q6_K.gguf", "ollama_tag": "llama4:scout-q6_K"},
    },
    # Llama 4 Maverick - MoE 401B total, 17B active, 128 experts, 1M context
    "llama4:maverick": {
        "UD-1.78bit": {"size_gb": 122.0, "vram_gb": 96.0, "quality": 2, "file": "llama4-maverick-ud-1.78bit", "ollama_tag": "llama4:maverick-ud"},
        "Q4_K_M": {"size_gb": 243.0, "vram_gb": 250.0, "quality": 3, "file": "Llama-4-Maverick-17B-128E-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "llama4:maverick"},
    },
    # Qwen3 8B - Efficient mid-size model with thinking mode
    "qwen3:8b": {
        "Q4_K_M": {"size_gb": 5.0, "vram_gb": 6.0, "quality": 3, "file": "Qwen_Qwen3-8B-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen3:8b"},
        "Q5_K_M": {"size_gb": 6.0, "vram_gb": 7.0, "quality": 4, "file": "Qwen_Qwen3-8B-Q5_K_M.gguf", "ollama_tag": "qwen3:8b-q5_K_M"},
        "Q8_0": {"size_gb": 8.5, "vram_gb": 10.0, "quality": 5, "file": "Qwen_Qwen3-8B-Q8_0.gguf", "ollama_tag": "qwen3:8b-q8_0"},
    },
    # Qwen3 32B - Dense model, strong all-rounder
    "qwen3:32b": {
        "Q4_K_M": {"size_gb": 19.8, "vram_gb": 22.0, "quality": 3, "file": "Qwen_Qwen3-32B-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen3:32b"},
        "Q5_K_M": {"size_gb": 24.0, "vram_gb": 26.0, "quality": 4, "file": "Qwen_Qwen3-32B-Q5_K_M.gguf", "ollama_tag": "qwen3:32b-q5_K_M"},
        "Q8_0": {"size_gb": 35.0, "vram_gb": 38.0, "quality": 5, "file": "Qwen_Qwen3-32B-Q8_0.gguf", "ollama_tag": "qwen3:32b-q8_0"},
    },
    # Qwen3-Coder 30B - MoE 30B total, 3.3B active, 256K context, tool calling fixed by Unsloth
    "qwen3-coder:30b": {
        "Q4_K_M": {"size_gb": 18.0, "vram_gb": 20.0, "quality": 3, "file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "default": True, "ollama_tag": "qwen3-coder:30b"},
        "Q5_K_M": {"size_gb": 22.0, "vram_gb": 24.0, "quality": 4, "file": "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf", "ollama_tag": "qwen3-coder:30b-q5_K_M"},
        "Q8_0": {"size_gb": 32.0, "vram_gb": 35.0, "quality": 5, "file": "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf", "ollama_tag": "qwen3-coder:30b-q8_0"},
    },
}

# HuggingFace repo mapping for GGUF downloads
HUGGINGFACE_REPOS = {
    "qwen2.5:7b": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "qwen2.5:14b": "bartowski/Qwen2.5-14B-Instruct-GGUF",
    "qwen2.5:32b": "bartowski/Qwen2.5-32B-Instruct-GGUF",
    "qwen2.5:72b": "bartowski/Qwen2.5-72B-Instruct-GGUF",
    "qwen2.5:3b": "bartowski/Qwen2.5-3B-Instruct-GGUF",
    "qwen3:32b": "bartowski/Qwen_Qwen3-32B-GGUF",
    "qwen3:8b": "bartowski/Qwen_Qwen3-8B-GGUF",
    "llama3.2:3b": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "llama3.1:8b": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "llama3.3:70b": "bartowski/Llama-3.3-70B-Instruct-GGUF",
    "mixtral:8x7b": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
}


async def detect_gpu_info() -> Dict[str, Any]:
    """
    Detect GPU information using nvidia-smi (NVIDIA) or system commands.
    Returns VRAM info, GPU name, and utilization.
    """
    gpu_info = {
        "detected": False,
        "gpus": [],
        "total_vram_gb": 0,
        "available_vram_gb": 0,
        "recommended_quant": "Q4_K_M",
        "max_model_size_gb": 0,
    }

    try:
        # Try nvidia-smi first (NVIDIA GPUs)
        # v0.3.3 Defect 18: wrap sync subprocess in asyncio.to_thread so a
        # hanging external process can't freeze the FastAPI event loop.
        result = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=_SUBPROCESS_FLAGS,
        )

        if result.returncode == 0:
            gpu_info["detected"] = True
            total_vram = 0
            available_vram = 0

            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        name = parts[0]
                        total_mb = float(parts[1])
                        free_mb = float(parts[2])
                        used_mb = float(parts[3])
                        util = parts[4] if len(parts) > 4 else "0"

                        total_gb = total_mb / 1024
                        free_gb = free_mb / 1024
                        used_gb = used_mb / 1024

                        gpu_info["gpus"].append({
                            "name": name,
                            "total_vram_gb": round(total_gb, 1),
                            "free_vram_gb": round(free_gb, 1),
                            "used_vram_gb": round(used_gb, 1),
                            "utilization_percent": int(util) if util.isdigit() else 0
                        })

                        total_vram += total_gb
                        available_vram += free_gb

            gpu_info["total_vram_gb"] = round(total_vram, 1)
            gpu_info["available_vram_gb"] = round(available_vram, 1)

            # Calculate max model size (leave ~2GB headroom for system)
            gpu_info["max_model_size_gb"] = max(0, round(available_vram - 2, 1))

            # Recommend quantization based on available VRAM
            if available_vram >= 48:
                gpu_info["recommended_quant"] = "Q8_0"  # Highest quality
            elif available_vram >= 24:
                gpu_info["recommended_quant"] = "Q6_K"
            elif available_vram >= 12:
                gpu_info["recommended_quant"] = "Q5_K_M"
            elif available_vram >= 8:
                gpu_info["recommended_quant"] = "Q4_K_M"  # Default balanced
            elif available_vram >= 4:
                gpu_info["recommended_quant"] = "Q3_K_M"
            else:
                gpu_info["recommended_quant"] = "Q2_K"  # Most compressed

    except FileNotFoundError:
        logger.debug("nvidia-smi not found - no NVIDIA GPU or drivers not installed")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    return gpu_info


def get_recommended_models(vram_gb: float) -> List[Dict[str, Any]]:
    """
    Get list of models that fit in available VRAM, sorted by quality.
    """
    recommendations = []

    for model_name, quants in QUANTIZATION_OPTIONS.items():
        for quant_level, info in quants.items():
            if info["vram_gb"] <= vram_gb:
                recommendations.append({
                    "model": model_name,
                    "quantization": quant_level,
                    "size_gb": info["size_gb"],
                    "vram_required_gb": info["vram_gb"],
                    "quality": info["quality"],
                    "is_default": info.get("default", False),
                    "headroom_gb": round(vram_gb - info["vram_gb"], 1)
                })

    # Sort by quality (descending), then by VRAM usage (descending to prefer larger models)
    recommendations.sort(key=lambda x: (-x["quality"], -x["vram_required_gb"]))

    return recommendations

# v0.3.3 Section 4H — tool capability is read from the dynamic capability
# cache populated by fetch_ollama_capabilities on registry refresh. No more
# hardcoded tier table that goes stale every time a new model ships.
#
# The "tier" concept is preserved only as informational metadata derived from
# the actual capabilities — verified=tools present, experimental=tools absent.
# Nothing gates runtime behavior on tier anymore.

def get_model_tier(model_name: str, provider: str) -> str:
    """Informational tier derived from cached capabilities.

    'verified' = model declares 'tools' capability via Ollama /api/show
    'experimental' = no tools capability OR cache miss (cold start)

    This is descriptive only — UI sort/badge hints. Tool-capability gating
    uses is_tool_capable directly against the cache.
    """
    return "verified" if is_tool_capable(model_name, provider) else "experimental"


def is_tool_capable(model_name: str, provider: str) -> bool:
    """Whether the model supports tool calling, from dynamic capability cache.

    Ollama: reads /api/show capabilities populated on registry refresh.
    LM Studio: defaults True (no metadata exposed in OpenAI-compatible API).
    Runtime tool-incompatibility detection in ollama_client.py retries
    without tools if the model 400s — false-positives self-heal.
    """
    return get_cached_tools(model_name, provider)

def get_model_description(model_name: str, tier: str) -> str:
    """Generate description based on model name and tier"""
    descriptions = {
        # Qwen 2.5 Series
        "qwen2.5:7b": "Tool calling may be unreliable",
        "qwen2.5:14b": "Larger Qwen - Great tool performance",
        "qwen2.5:32b": "Powerful Qwen - Excellent tools",
        "qwen2.5:72b": "Massive Qwen - Superior tool calling",
        "qwen2.5:3b": "May have inconsistent tool calling",

        # Llama 3 Series
        "llama3.3:70b": "Meta's latest 70B - Native tools, 128K context",
        "llama3.1:70b": "Meta's flagship - Excellent tools, 128K context",
        "llama3.1:8b": "Unreliable tool calling",
        "llama3.2:8b": "Latest compact Llama - Tool support",
        "llama3.2:3b": "May output JSON instead of calling tools",

        # Llama 4 Series - MoE with native tool calling
        "llama4:scout": "MoE 109B - 10M context, native tools",
        "llama4:maverick": "MoE 401B - 128 experts, 1M context, native tools",

        # OpenAI Open Source
        "gpt-oss:20b": "OpenAI efficient model - Excellent tools",
        "gpt-oss:120b": "OpenAI flagship open model - Native tools",

        # Qwen3 Series
        "qwen3:32b": "Alibaba flagship - Native Hermes tools",
        "qwen3:8b": "Efficient Qwen3 - Native tools",
        "qwen3:4b": "Compact Qwen3 - Native tools",
        "qwen3:14b": "Mid-size Qwen3 - Native tools",
        "qwen3-coder:30b": "MoE 30B - 256K context, tool calling (Unsloth fixed)",

        # Mistral Family
        "mistral:7b": "Fast and efficient with tools",
        "mixtral:8x7b": "Mixture of Experts - Native tools",

        # Other
        "command-r": "Cohere's command model - Strong tools",
        "phi-4": "Microsoft's efficient model",
        "phi3": "Compact Microsoft model",

        # Experimental (known issues)
        "deepseek-r1": "Reasoning model - No tool support",
        "deepseek-v3": "MoE 671B - Unstable tool calling",
        "gemma2:9b": "Google - No native tool support",
        "gemma2:27b": "Google - No native tool support",
    }

    # Get base description
    desc = None
    model_lower = model_name.lower()
    for key, description in descriptions.items():
        if key in model_lower:
            desc = description
            break

    if not desc:
        desc = "Custom model"

    # Add tier indicator
    if tier == "verified":
        return f"✅ {desc}"
    elif tier == "compatible":
        return f"⚠️ {desc} (untested)"
    else:
        return f"🧪 {desc} (experimental - may not work with tools)"

def extract_quantization(file_name: str) -> Optional[str]:
    """Extract quantization level from GGUF filename"""
    if not file_name or "Q" not in file_name:
        return None

    # Look for patterns like Q4_K_M, Q5_K_S, Q8_0, etc.
    import re
    match = re.search(r'Q(\d+)_([A-Z0-9_]+)', file_name)
    if match:
        return f"Q{match.group(1)}_{match.group(2)}"

    return None

@router.get("/api/model/registry")
async def get_model_registry(
    provider: Optional[str] = None,
    tool_capable_only: bool = True
) -> Dict[str, Any]:
    """
    Get unified model registry with metadata from all config sources.

    Query Parameters:
    - provider: Filter by provider (ollama/lmstudio)
    - tool_capable_only: Only return models with reliable tool support (default: True)

    Returns:
    - models: List of model metadata
    - count: Total number of models
    - providers: List of available providers
    """
    from app.routers.model_switcher import MODEL_TO_HUGGINGFACE, PROVIDERS, detect_provider

    registry = []
    available_providers = []

    # Detect which providers are running
    for provider_name, provider_config in PROVIDERS.items():
        if provider and provider != provider_name:
            continue

        try:
            provider_info = await detect_provider(provider_name, provider_config)
            if not provider_info:
                logger.debug(f"Provider {provider_name} not available")
                continue

            available_providers.append(provider_name)

            # Get models from provider
            if provider_name == "ollama":
                # Use Ollama API to get installed models
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"http://localhost:{provider_config['port']}/api/tags")
                        if response.status_code == 200:
                            data = response.json()
                            models = [m['name'] for m in data.get('models', [])]
                        else:
                            models = []
                except Exception as e:
                    logger.error(f"Failed to fetch Ollama models: {e}")
                    models = []

            elif provider_name == "lmstudio":
                # Use LM Studio API to get loaded models
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"http://localhost:{provider_config['port']}/v1/models")
                        if response.status_code == 200:
                            data = response.json()
                            models = [m['id'] for m in data.get('data', [])]
                        else:
                            models = []
                except Exception as e:
                    logger.error(f"Failed to fetch LM Studio models: {e}")
                    models = []
            else:
                models = []

            # v0.3.3 §4H — populate capability cache BEFORE the per-model loop,
            # so the sync get_cached_* readers below have fresh data.
            #
            # Ollama exposes capabilities via /api/show, so a single GET per
            # model is enough.
            #
            # LM Studio's OpenAI-compatible /v1/models does NOT expose
            # capabilities, so we fire probe_vision (1x1 PNG canary) per model.
            # probe_vision is cache-aware (24h TTL) — warm path returns
            # immediately without a network call, cold path actually probes.
            if provider_name == "ollama" and models:
                import asyncio
                base_url = f"http://localhost:{provider_config['port']}"
                eligible = [
                    m for m in models
                    if not any(x in m.lower() for x in ['embed', 'nomic', 'bge-', 'minilm'])
                ]
                await asyncio.gather(
                    *(fetch_ollama_capabilities(m, base_url) for m in eligible),
                    return_exceptions=True,
                )
            elif provider_name == "lmstudio" and models:
                # v0.3.3 Defect 9: prefer LM Studio's native `/api/v0/models`
                # extension over the canary probe. The OpenAI-spec `/v1/models`
                # does NOT expose capabilities, but LM Studio's richer endpoint
                # returns `type == "vlm"` for Vision-Language Models and lists
                # `tool_use` in `capabilities`. This is instant, deterministic,
                # and works even when the model is in `state: "loading"` (the
                # canary probe needs the model fully loaded and times out at
                # 10 s, leaving large vision models like qwen3.6-35b-a3b
                # permanently classified as `vision: false` until they finish
                # warming up). Fall back to the canary probe only when the
                # extended endpoint isn't available (older LM Studio builds).
                import asyncio
                import httpx
                base_url = f"http://localhost:{provider_config['port']}"
                eligible = [
                    m for m in models
                    if not any(x in m.lower() for x in ['embed', 'nomic', 'bge-', 'minilm'])
                ]
                used_extended_api = False
                try:
                    async with httpx.AsyncClient(timeout=3.0) as caps_client:
                        v0_resp = await caps_client.get(f"{base_url}/api/v0/models")
                    if v0_resp.status_code == 200:
                        v0_lookup = {
                            m.get("id"): m for m in v0_resp.json().get("data", [])
                        }
                        for m in eligible:
                            meta = v0_lookup.get(m)
                            if not meta:
                                continue
                            # Always include "tools" for LM Studio to preserve
                            # the existing default-True behavior (see
                            # get_cached_tools comment in model_contexts.py).
                            # Without this, writing a cache entry that omits
                            # "tools" would flip tool-capable models to False
                            # on older LM Studio builds where /api/v0/models
                            # doesn't list tool_use. Runtime retry-on-400 in
                            # ollama_client already self-heals false positives.
                            caps_set = {"tools"}
                            if meta.get("type") == "vlm":
                                caps_set.add("vision")
                            _save_caps_cache(
                                _caps_cache_key(m, "lmstudio"), caps_set
                            )
                        used_extended_api = True
                        logger.info(
                            f"[REGISTRY] Populated LM Studio caps from "
                            f"/api/v0/models for {len(eligible)} model(s)"
                        )
                except (httpx.HTTPError, ValueError, TypeError) as caps_err:
                    logger.debug(
                        f"[REGISTRY] /api/v0/models unavailable, falling back "
                        f"to canary probe: {caps_err}"
                    )
                if not used_extended_api:
                    await asyncio.gather(
                        *(probe_vision(m, "lmstudio", base_url) for m in eligible),
                        return_exceptions=True,
                    )

            # Process each model
            for model_name in models:
                # Skip non-chat models (embeddings only) - multimodal models like llava are valid chat APIs now
                if any(x in model_name.lower() for x in ['embed', 'nomic', 'bge-', 'minilm']):
                    continue

                # Check tool capability (reads dynamic capability cache)
                tier = get_model_tier(model_name, provider_name)
                if tool_capable_only and not is_tool_capable(model_name, provider_name):
                    continue

                # Get context window
                context_size = get_context_size(model_name)

                # Get GGUF info if available
                gguf_info = MODEL_TO_HUGGINGFACE.get(model_name, {})
                size_gb = gguf_info.get("size_gb")
                quantization = extract_quantization(gguf_info.get("file", ""))

                # Get max context from MODEL_CONTEXTS
                model_family = model_name.split(':')[0]
                max_context = MODEL_CONTEXTS.get(model_family, {}).get("max", context_size)

                registry.append({
                    "name": model_name,
                    "provider": provider_name,
                    "tier": tier,
                    "size_gb": size_gb,
                    "context_window": context_size,
                    "max_context": max_context,
                    "quantization": quantization,
                    "capabilities": {
                        "tools": is_tool_capable(model_name, provider_name),
                        "streaming": True,
                        "vision": get_cached_vision(model_name, provider_name)
                    },
                    "description": get_model_description(model_name, tier)
                })

        except Exception as e:
            logger.error(f"Error processing provider {provider_name}: {e}", exc_info=True)
            continue

    return {
        "models": registry,
        "count": len(registry),
        "providers": available_providers,
    }


@router.post("/api/model/probe_vision/{model_name}")
async def probe_model_vision(model_name: str) -> Dict[str, Any]:
    """Run a vision capability probe for a specific model.

    Uses the async canary-image method with disk caching.
    Returns immediately if a valid cached result exists (< 24h old).
    """
    from app.routers.model_switcher import PROVIDERS, detect_provider
    from config.model_contexts import (
        _caps_cache_key,
        _load_caps_cache,
        probe_vision as _probe_vision,
    )

    provider_info = None
    for pname, pconfig in PROVIDERS.items():
        info = await detect_provider(pname, pconfig)
        if info:
            provider_info = (pname, f"http://localhost:{pconfig['port']}")
            break

    if not provider_info:
        raise HTTPException(status_code=503, detail="No model provider available")

    # Check capability cache before probing so we can report whether the
    # result came from cache (v0.3.3 §4H: unified caps cache replaces the
    # old vision-only cache).
    cache_key = _caps_cache_key(model_name, provider_info[0])
    was_cached = bool(_load_caps_cache().get(cache_key))

    result = await _probe_vision(model_name, provider_info[0], provider_info[1])

    return {
        "model": model_name,
        "vision_capable": result,
        "cached": was_cached and result is not None,
    }


@router.get("/api/model/tiers")
async def get_model_tiers() -> Dict[str, Any]:
    """v0.3.3 §4H deprecation: tier groupings are no longer hardcoded.

    Tier is now derived per-model from dynamic Ollama /api/show capabilities
    (see is_tool_capable in this file). The frontend should read each
    installed model's `tier` field from `/api/model/registry` directly
    instead of consulting a static tier→[models] map.
    """
    return {
        "deprecated": True,
        "message": (
            "Hardcoded tier groupings were removed in v0.3.3. "
            "Read per-model `tier` from /api/model/registry instead."
        ),
        "tiers": {},
    }


# ============================================================================
# NEW: GPU/VRAM Detection and Model Recommendations
# ============================================================================

@router.get("/api/model/gpu")
async def get_gpu_info() -> Dict[str, Any]:
    """
    Detect GPU information and available VRAM.

    Returns:
    - detected: Whether a GPU was found
    - gpus: List of detected GPUs with VRAM info
    - total_vram_gb: Total VRAM across all GPUs
    - available_vram_gb: Free VRAM currently available
    - recommended_quant: Suggested quantization based on VRAM
    - max_model_size_gb: Largest model that fits (with headroom)
    """
    return await detect_gpu_info()


@router.get("/api/model/catalog")
async def get_model_catalog() -> Dict[str, Any]:
    """
    Get full model catalog with all quantization options.

    Returns comprehensive catalog of available models with:
    - All quantization levels and their VRAM requirements
    - File sizes and quality ratings
    - HuggingFace repo information for downloads
    """
    catalog = []

    for model_name, quants in QUANTIZATION_OPTIONS.items():
        repo = HUGGINGFACE_REPOS.get(model_name)
        context_size = get_context_size(model_name)

        # v0.3.3 §4H — catalog models are pre-install candidates with no
        # running provider to query. Tier/tool_capable are intentionally
        # absent here; the frontend should fetch the authoritative values
        # from /api/model/registry after the user installs the model.
        model_entry = {
            "name": model_name,
            "tier": "unknown",
            "tool_capable": None,
            "context_window": context_size,
            "huggingface_repo": repo,
            "quantizations": []
        }

        for quant_level, info in quants.items():
            model_entry["quantizations"].append({
                "level": quant_level,
                "size_gb": info["size_gb"],
                "vram_required_gb": info["vram_gb"],
                "quality": info["quality"],
                "quality_label": ["", "Low", "Medium-Low", "Balanced", "High", "Highest"][info["quality"]],
                "file": info["file"],
                "is_default": info.get("default", False),
                "download_url": f"https://huggingface.co/{repo}/resolve/main/{info['file']}" if repo else None
            })

        # Sort quantizations by quality descending
        model_entry["quantizations"].sort(key=lambda x: -x["quality"])
        catalog.append(model_entry)

    return {
        "models": catalog,
        "count": len(catalog)
    }


@router.get("/api/model/recommendations")
async def get_model_recommendations(
    vram_gb: Optional[float] = None,
    include_all: bool = False
) -> Dict[str, Any]:
    """
    Get model recommendations based on available VRAM.

    Query Parameters:
    - vram_gb: Override VRAM detection with manual value (optional)
    - include_all: Include models that don't fit in VRAM (default: False)

    Returns:
    - gpu_info: Detected GPU information
    - recommendations: Models sorted by quality that fit in VRAM
    - warnings: Any warnings about model selection
    """
    # Detect GPU if vram_gb not specified
    gpu_info = await detect_gpu_info()

    effective_vram = vram_gb if vram_gb is not None else gpu_info["available_vram_gb"]

    recommendations = get_recommended_models(effective_vram if not include_all else 999)

    # Add warnings
    warnings = []
    if not gpu_info["detected"]:
        warnings.append("No NVIDIA GPU detected. Models will run on CPU (slower).")
    if effective_vram < 4:
        warnings.append("Limited VRAM. Consider Q2_K or Q3_K_M quantizations for better performance.")
    if effective_vram < 2:
        warnings.append("Very limited VRAM. CPU offloading will be required for most models.")

    # Group by model for cleaner output
    by_model = {}
    for rec in recommendations:
        model = rec["model"]
        if model not in by_model:
            by_model[model] = {
                "model": model,
                # v0.3.3 §4H: pre-install recommendations have no provider to
                # query; tier/tool_capable are populated on /api/model/registry
                # once the user actually installs the model.
                "tier": "unknown",
                "tool_capable": None,
                "available_quantizations": []
            }
        by_model[model]["available_quantizations"].append({
            "quantization": rec["quantization"],
            "size_gb": rec["size_gb"],
            "vram_required_gb": rec["vram_required_gb"],
            "quality": rec["quality"],
            "is_default": rec["is_default"],
            "headroom_gb": rec["headroom_gb"]
        })

    return {
        "gpu_info": gpu_info,
        "effective_vram_gb": effective_vram,
        "recommendations": list(by_model.values()),
        "total_options": len(recommendations),
        "warnings": warnings
    }


@router.get("/api/model/{model_name}/quantizations")
async def get_model_quantizations(model_name: str) -> Dict[str, Any]:
    """
    Get all available quantizations for a specific model.

    Path Parameters:
    - model_name: Model name (e.g., "qwen2.5:7b")

    Returns quantization options with VRAM requirements.
    """
    if model_name not in QUANTIZATION_OPTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in quantization catalog. Available: {list(QUANTIZATION_OPTIONS.keys())}"
        )

    quants = QUANTIZATION_OPTIONS[model_name]
    repo = HUGGINGFACE_REPOS.get(model_name)

    # Get GPU info for recommendations
    gpu_info = await detect_gpu_info()

    options = []
    for quant_level, info in quants.items():
        fits_in_vram = info["vram_gb"] <= gpu_info["total_vram_gb"] if gpu_info["detected"] else True
        options.append({
            "level": quant_level,
            "size_gb": info["size_gb"],
            "vram_required_gb": info["vram_gb"],
            "quality": info["quality"],
            "quality_label": ["", "Low", "Medium-Low", "Balanced", "High", "Highest"][info["quality"]],
            "file": info["file"],
            "is_default": info.get("default", False),
            "fits_in_vram": fits_in_vram,
            "download_url": f"https://huggingface.co/{repo}/resolve/main/{info['file']}" if repo else None,
            "ollama_tag": info.get("ollama_tag", model_name)  # Ollama tag for pulling this quantization
        })

    # Sort by quality descending
    options.sort(key=lambda x: -x["quality"])

    return {
        "model": model_name,
        "huggingface_repo": repo,
        "quantizations": options,
        "gpu_info": gpu_info,
        "recommended_quant": gpu_info["recommended_quant"]
    }
