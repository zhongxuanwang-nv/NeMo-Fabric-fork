# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric model aliases mapped to LangChain chat models.

The provider scope matches the Deep Agents adapter so both LangChain-family
adapters resolve a Fabric ``models`` entry the same way: NVIDIA-hosted and
OpenAI-compatible endpoints go through ``ChatOpenAI``, and any other provider
falls through to ``init_chat_model``.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils

from nemo_fabric_adapters.langgraph.config import AdapterConfigError

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Providers served through the OpenAI-compatible ``ChatOpenAI`` client.
OPENAI_COMPATIBLE_PROVIDERS = {"", "nvidia", "openai", "openai-compatible"}
# Providers whose default endpoint is NVIDIA's OpenAI-compatible gateway.
NVIDIA_DEFAULT_PROVIDERS = {"", "nvidia"}
# Conventional credential env var per provider; others must set api_key_env.
PROVIDER_DEFAULT_API_KEY_ENV = {
    "": "NVIDIA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def model_config(payload: dict[str, Any], alias: str) -> dict[str, Any]:
    models = common_utils.models_payload(payload)
    config = models.get(alias)
    if not isinstance(config, dict):
        available = sorted(models) or ["<none>"]
        raise AdapterConfigError(f"models.{alias} is not configured; configured aliases are {available}.")
    return config


def resolve_api_key_env(config: dict[str, Any]) -> str:
    """Resolve the credential env var, defaulting per provider.

    An explicit ``api_key_env`` always wins. Otherwise nvidia/unspecified default to
    ``NVIDIA_API_KEY`` and openai to ``OPENAI_API_KEY``; any other provider must set
    ``api_key_env`` explicitly so a key is never sent to the wrong endpoint.
    """

    explicit = config.get("api_key_env")
    if explicit:
        return str(explicit)
    provider = str(config.get("provider") or "").lower()
    default = PROVIDER_DEFAULT_API_KEY_ENV.get(provider)
    if default is None:
        raise AdapterConfigError(f"models.<alias>.api_key_env is required for provider '{provider}'.")
    return default


def resolve_base_url(config: dict[str, Any]) -> str | None:
    base_url = (config.get("settings") or {}).get("base_url") or config.get("base_url")
    if base_url:
        return str(base_url)
    # Only NVIDIA (or an unspecified provider) defaults to NVIDIA's endpoint; a plain
    # ``openai`` provider falls through to the ChatOpenAI default.
    if str(config.get("provider") or "").lower() in NVIDIA_DEFAULT_PROVIDERS:
        return DEFAULT_NVIDIA_BASE_URL
    return None


def build_chat_model(payload: dict[str, Any], alias: str) -> tuple[Any, dict[str, Any]]:
    """Build a LangChain chat model for a Fabric model alias.

    Returns the model and a JSON-safe descriptor of what was bound, which the
    adapter reports as result metadata.
    """

    config = model_config(payload, alias)
    model_name = config.get("model")
    if not model_name:
        raise AdapterConfigError(f"models.{alias}.model is required.")

    api_key_env = resolve_api_key_env(config)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise AdapterConfigError(
            f"the model-provider credential env var '{api_key_env}' is not set in the environment. Set it "
            f"to your API key, or set models.{alias}.api_key_env to the variable that holds it."
        )

    provider = str(config.get("provider") or "nvidia").lower()
    base_url = resolve_base_url(config)
    temperature = (config.get("settings") or {}).get("temperature", config.get("temperature"))
    descriptor = {"alias": alias, "model": str(model_name), "provider": provider, "base_url": base_url}

    kwargs: dict[str, Any] = {"model": model_name, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if temperature is not None:
        kwargs["temperature"] = temperature

    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        # Generic provider hook: honor an explicit non-OpenAI-compatible provider.
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:
            raise AdapterConfigError(_models_extra_error(provider)) from exc

        kwargs["model_provider"] = provider
        return init_chat_model(**_supported_kwargs(init_chat_model, kwargs)), descriptor

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise AdapterConfigError(_models_extra_error(provider)) from exc

    return ChatOpenAI(**_supported_kwargs(ChatOpenAI, kwargs)), descriptor


def _models_extra_error(provider: str) -> str:
    return (
        f"a Fabric-managed chat model was requested for provider '{provider}' but the LangChain model "
        "clients are not installed; install the adapter's 'models' extra "
        "(pip install 'nemo-fabric-adapters-langgraph[models]')."
    )


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters.values()
    if any(param.kind == param.VAR_KEYWORD for param in parameters):
        return dict(kwargs)
    supported = {param.name for param in parameters}
    return {key: value for key, value in kwargs.items() if key in supported}
