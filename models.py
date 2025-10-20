from typing import Set

import httpx

from .exceptions import G4FProException, G4FProTimeoutError, APIError, G4FProParseError, G4FProConnectionError


class Models:
    """
    Provides access to the available model list from the G4FPro API.

    Includes both synchronous and asynchronous methods
    to fetch and filter supported models (chat, image, etc.).
    """

    URL: str = "https://gpt4free.pro/v1/models"

    CHAT_MODELS: Set[str] = {
        "deepseek-v3.1", "command-a", "deepseek-chat", "deepseek-reasoner", "glm-4.6",
        "gpt-5-nano", "claude-sonnet-4", "claude-3-7-sonnet", "claude-sonnet-4.5",
        "claude-haiku-4.5", "hermes-4-405b", "hermes-3-405b", "qwen3-coder", "qwen3-coder-big",
        "qwq-32b-fast", "gpt-oss-120b", "llama-3.3", "kimi-k2", "kimi-k2-0905", "llama-4-maverick",
        "llama-4-scout", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
        "lucid-origin", "grok-4", "grok-4-think", "grok-code-1", "grok-3-mini", "gpt-5-chat",
        "gpt-5-mini", "qwen2.5-coder-32b", "qwen3-omni", "qwen3-next", "deepseek-r1-0528",
        "mistral-small-3.1-24b", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "o3-mini",
        "gpt-4o-mini", "gpt-3.5-turbo", "deepseek-v3", "deepseek-v3.2", "deepseek-r1",
        "glm-4.5", "glm-4.5-air", "ring-1t", "ling-1t", "ernie-4.5", "sonar"
    }

    IMAGE_MODELS: Set[str] = {
        "flux-schnell", "sdxl", "nano-banana", "gpt-image-1"
    }

    ALL_MODELS: Set[str] = CHAT_MODELS | IMAGE_MODELS

    @staticmethod
    def _fetch_remote_models() -> Set[str]:
        """
        Fetches available models from G4FPro API (synchronously).
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(Models.URL)

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    message = error_data.get("error", {}).get("message", response.text)
                except Exception:
                    message = response.text
                raise APIError(status_code=response.status_code, message=message)

            data = response.json()
            return {model["id"] for model in data.get("data", [])}

        except httpx.TimeoutException:
            raise G4FProTimeoutError("Request to G4FPro API timed out.")
        except httpx.RequestError as e:
            raise G4FProConnectionError(f"Network error while requesting G4FPro API: {e}")
        except ValueError as e:
            raise G4FProParseError(f"Invalid JSON in G4FPro API response: {e}")
        except Exception as e:
            raise G4FProException(f"Unexpected error while fetching models: {e}")

    @staticmethod
    async def _fetch_remote_models_async() -> Set[str]:
        """
        Fetches available models from G4FPro API (asynchronously).
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(Models.URL)
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    message = error_data.get("error", {}).get("message", response.text)
                except Exception:
                    message = response.text
                raise APIError(status_code=response.status_code, message=message)

            data = response.json()
            return {model["id"] for model in data.get("data", [])}

        except httpx.TimeoutException:
            raise G4FProTimeoutError("Request to G4FPro API timed out.")
        except httpx.RequestError as e:
            raise G4FProConnectionError(f"Network error while requesting G4FPro API: {e}")
        except ValueError as e:
            raise G4FProParseError(f"Invalid JSON in G4FPro API response: {e}")
        except Exception as e:
            raise G4FProException(f"Unexpected error while fetching models: {e}")

    @classmethod
    def get_all_models(cls) -> Set[str]:
        """Returns all supported models available on the server."""
        return cls.ALL_MODELS & cls._fetch_remote_models()

    @classmethod
    def get_chat_models(cls) -> Set[str]:
        """Returns all available chat-capable models."""
        return cls.CHAT_MODELS & cls._fetch_remote_models()

    @classmethod
    def get_image_models(cls) -> Set[str]:
        """Returns all available image-generation models."""
        return cls.IMAGE_MODELS & cls._fetch_remote_models()

    @classmethod
    async def get_all_models_async(cls) -> Set[str]:
        """Asynchronously returns all supported models available on the server."""
        remote_models = await cls._fetch_remote_models_async()
        return cls.ALL_MODELS & remote_models

    @classmethod
    async def get_chat_models_async(cls) -> Set[str]:
        """Asynchronously returns all available chat-capable models."""
        remote_models = await cls._fetch_remote_models_async()
        return cls.CHAT_MODELS & remote_models

    @classmethod
    async def get_image_models_async(cls) -> Set[str]:
        """Asynchronously returns all available image-generation models."""
        remote_models = await cls._fetch_remote_models_async()
        return cls.IMAGE_MODELS & remote_models