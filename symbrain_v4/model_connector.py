#!/usr/bin/env python3
"""
SymBrain v4 — Production Model Connector
==========================================

Connects the v4 inference server to real model backends:
  • Local Ollama models (7B tier on Apple Silicon / edge devices)
  • HuggingFace Transformers (32B tier on Cloud Run with L4 GPU)
  • vLLM / TGI remote backends (70B/122B tiers on multi-GPU)

This module replaces the SimulationEngine in production mode.

(c) 2026 Socrate AI Lab, Paris, France
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger("symbrain.v4.connector")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Abstract Backend                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class ModelBackend(ABC):
    """Abstract interface for model inference backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the backend is available and responsive."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model identifier."""
        ...


# ═══════════════════════════════════════════════════════════════════════════ #
#  Ollama Backend (Edge / Local)                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class OllamaBackend(ModelBackend):
    """Connect to a local Ollama server for edge-tier inference.

    Ollama serves quantized models efficiently on Apple Silicon
    using Metal MPS acceleration with zero-copy UMA memory.

    Default models:
        Deductive:  mistral:7b-instruct  (Mistral 7B, GGUF Q4_K_M)
        Generative: gemma2:latest        (Gemma 2 9B, GGUF Q4_K_M)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        deductive_model: str = "mistral:7b-instruct",
        generative_model: str = "gemma2:latest",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._deductive = deductive_model
        self._generative = generative_model
        self._client = httpx.Client(timeout=timeout)
        logger.info(
            "OllamaBackend initialized: %s (ded=%s, gen=%s)",
            base_url, deductive_model, generative_model,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate via Ollama /api/generate endpoint."""
        payload = {
            "model": self._deductive,
            "prompt": prompt,
            "system": system_prompt or self._default_system_prompt(),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.95,
                "repeat_penalty": 1.1,
            },
        }

        try:
            resp = self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except httpx.HTTPError as e:
            logger.error("Ollama generation failed: %s", e)
            raise RuntimeError(f"Ollama backend error: {e}") from e

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate via Ollama /api/chat endpoint (multi-turn)."""
        payload = {
            "model": self._deductive,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            logger.error("Ollama chat failed: %s", e)
            raise RuntimeError(f"Ollama backend error: {e}") from e

    def is_healthy(self) -> bool:
        try:
            resp = self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def model_name(self) -> str:
        return f"ollama/{self._deductive}"

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are SymBrain v4, a neurosymbolic AI mathematician and scientist. "
            "You solve problems with rigorous formal reasoning, showing complete "
            "step-by-step derivations. Always verify your answers. Use mathematical "
            "notation precisely. For French language problems, respond in French "
            "with proper mathematical terminology."
        )

    def list_models(self) -> list[str]:
        """Return available Ollama models."""
        try:
            resp = self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════ #
#  OpenAI-Compatible Backend (vLLM / TGI / LiteLLM)                          #
# ═══════════════════════════════════════════════════════════════════════════ #

class OpenAICompatibleBackend(ModelBackend):
    """Connect to any OpenAI-compatible API (vLLM, TGI, LiteLLM, etc.).

    This backend supports the 32B/70B/122B cloud tiers where models
    are served via vLLM with tensor parallelism on A100/H100 GPUs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-Math-32B-Instruct",
        api_key: str = "EMPTY",
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)
        logger.info(
            "OpenAICompatibleBackend initialized: %s (model=%s)",
            base_url, model,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate via OpenAI-compatible /chat/completions endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": self._default_system_prompt(),
            })
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
        }

        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        except httpx.HTTPError as e:
            logger.error("OpenAI-compatible generation failed: %s", e)
            raise RuntimeError(f"Backend error: {e}") from e

    def is_healthy(self) -> bool:
        try:
            resp = self._client.get(f"{self._base_url}/models")
            return resp.status_code == 200
        except Exception:
            return False

    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are SymBrain v4, a neurosymbolic AI system for formal "
            "mathematical and scientific reasoning. Your Calibrated PFC "
            "(Prefrontal Cortex) router has classified this query as requiring "
            "rigorous deductive reasoning.\n\n"
            "Instructions:\n"
            "1. Show complete step-by-step derivations\n"
            "2. Use precise mathematical notation\n"
            "3. Verify your final answer\n"
            "4. Cite relevant theorems and lemmas\n"
            "5. For French-language problems, respond in French\n"
            "6. Mark your final answer clearly with **bold**"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Swarm Connector (Multi-Backend Orchestrator)                               #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class SwarmConfig:
    """Configuration for the production model swarm."""
    edge_ollama_url: str = "http://localhost:11434"
    edge_deductive_model: str = "mistral:7b-instruct"
    edge_generative_model: str = "gemma2:latest"

    cloud_32b_url: str = "http://localhost:8000/v1"
    cloud_32b_model: str = "Qwen/Qwen2.5-Math-32B-Instruct"
    cloud_32b_api_key: str = "EMPTY"

    cloud_70b_url: str = ""
    cloud_70b_model: str = "deepseek-ai/DeepSeek-Math-70B"
    cloud_70b_api_key: str = "EMPTY"

    cloud_122b_url: str = ""
    cloud_122b_model: str = "mistralai/Mistral-Large-Instruct-2"
    cloud_122b_api_key: str = "EMPTY"

    @classmethod
    def from_env(cls) -> "SwarmConfig":
        """Load configuration from environment variables."""
        return cls(
            edge_ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            edge_deductive_model=os.getenv("EDGE_DEDUCTIVE_MODEL", "mistral:7b-instruct"),
            edge_generative_model=os.getenv("EDGE_GENERATIVE_MODEL", "gemma2:latest"),
            cloud_32b_url=os.getenv("CLOUD_32B_URL", "http://localhost:8000/v1"),
            cloud_32b_model=os.getenv("CLOUD_32B_MODEL", "Qwen/Qwen2.5-Math-32B-Instruct"),
            cloud_32b_api_key=os.getenv("CLOUD_32B_API_KEY", "EMPTY"),
            cloud_70b_url=os.getenv("CLOUD_70B_URL", ""),
            cloud_70b_model=os.getenv("CLOUD_70B_MODEL", "deepseek-ai/DeepSeek-Math-70B"),
            cloud_70b_api_key=os.getenv("CLOUD_70B_API_KEY", "EMPTY"),
            cloud_122b_url=os.getenv("CLOUD_122B_URL", ""),
            cloud_122b_model=os.getenv("CLOUD_122B_MODEL", "mistralai/Mistral-Large-Instruct-2"),
            cloud_122b_api_key=os.getenv("CLOUD_122B_API_KEY", "EMPTY"),
        )


class ProductionSwarm:
    """Multi-backend model connector for the SymBrain v4 swarm.

    Maps model tiers to backend instances:
        EDGE_7B   → OllamaBackend (local, Apple Silicon / Jetson)
        CLOUD_32B → OpenAICompatibleBackend (Cloud Run + L4 GPU)
        CLOUD_70B → OpenAICompatibleBackend (GCE + 2×A100)
        CLOUD_122B → OpenAICompatibleBackend (GCE + 4×A100)
    """

    def __init__(self, config: SwarmConfig | None = None) -> None:
        self._config = config or SwarmConfig.from_env()
        self._backends: dict[str, ModelBackend] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize available backends."""
        cfg = self._config

        # Edge tier (always attempt)
        self._backends["7B"] = OllamaBackend(
            base_url=cfg.edge_ollama_url,
            deductive_model=cfg.edge_deductive_model,
            generative_model=cfg.edge_generative_model,
        )

        # Cloud 32B
        if cfg.cloud_32b_url:
            self._backends["32B"] = OpenAICompatibleBackend(
                base_url=cfg.cloud_32b_url,
                model=cfg.cloud_32b_model,
                api_key=cfg.cloud_32b_api_key,
            )

        # Cloud 70B
        if cfg.cloud_70b_url:
            self._backends["70B"] = OpenAICompatibleBackend(
                base_url=cfg.cloud_70b_url,
                model=cfg.cloud_70b_model,
                api_key=cfg.cloud_70b_api_key,
            )

        # Cloud 122B
        if cfg.cloud_122b_url:
            self._backends["122B"] = OpenAICompatibleBackend(
                base_url=cfg.cloud_122b_url,
                model=cfg.cloud_122b_model,
                api_key=cfg.cloud_122b_api_key,
            )

        available = [
            tier for tier, backend in self._backends.items()
            if backend.is_healthy()
        ]
        logger.info(
            "ProductionSwarm initialized: %d backends, %d healthy: %s",
            len(self._backends), len(available), available,
        )

    def get_backend(self, tier: str) -> ModelBackend | None:
        """Get the backend for a specific tier."""
        return self._backends.get(tier)

    def get_best_available(self, preferred_tier: str = "32B") -> ModelBackend | None:
        """Get the best available backend, preferring the specified tier."""
        # Try preferred tier first
        backend = self._backends.get(preferred_tier)
        if backend and backend.is_healthy():
            return backend

        # Fall back to largest healthy backend
        for tier in ["122B", "70B", "32B", "7B"]:
            backend = self._backends.get(tier)
            if backend and backend.is_healthy():
                logger.info("Falling back from %s to %s tier", preferred_tier, tier)
                return backend

        return None

    def generate(
        self,
        prompt: str,
        tier: str = "32B",
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> tuple[str, str]:
        """Generate using the specified tier, with automatic fallback.

        Returns (response, actual_tier_used).
        """
        backend = self.get_best_available(tier)
        if backend is None:
            raise RuntimeError("No healthy model backends available")

        response = backend.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        actual_tier = next(
            (t for t, b in self._backends.items() if b is backend),
            "unknown",
        )
        return response, actual_tier

    def health_report(self) -> dict[str, Any]:
        """Return health status of all backends."""
        report = {}
        for tier, backend in self._backends.items():
            healthy = backend.is_healthy()
            report[tier] = {
                "model": backend.model_name(),
                "healthy": healthy,
                "type": type(backend).__name__,
            }
        return report


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI Test                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def _test_connector() -> None:
    """Test the production model connector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
    )

    print("=" * 76)
    print("  SYMBRAIN v4 — PRODUCTION MODEL CONNECTOR TEST")
    print("=" * 76)

    config = SwarmConfig.from_env()
    swarm = ProductionSwarm(config)

    print("\n  Backend Health Report:")
    for tier, status in swarm.health_report().items():
        icon = "✅" if status["healthy"] else "❌"
        print(f"    {icon} {tier}: {status['model']} ({status['type']})")

    # Test with Ollama if available
    ollama = swarm.get_backend("7B")
    if ollama and ollama.is_healthy():
        print("\n  Testing Ollama 7B backend...")
        try:
            t0 = time.time()
            response = ollama.generate(
                "Calculate lim_{x→0} sin(x)/x. Show your work.",
                max_tokens=512,
                temperature=0.3,
            )
            elapsed = time.time() - t0
            print(f"    ✅ Response ({elapsed:.1f}s):")
            print(f"    {response[:300]}...")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    else:
        print("\n  ⚠️ Ollama not available — skipping live test")

    print("\n" + "=" * 76)


if __name__ == "__main__":
    _test_connector()
