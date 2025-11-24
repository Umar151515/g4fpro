from typing import Set

import httpx

from .http_client import sync_client, async_client
from .exceptions import G4FProException, G4FProTimeoutError, APIError, G4FProParseError, G4FProConnectionError


class Models:
    """
    Provides access to the available model list from the G4FPro API.

    Includes both synchronous and asynchronous methods
    to fetch and filter supported models (chat, image, etc.).
    """

    URL: str = "https://gpt4free.pro/v1/models"

    CHAT_MODELS: Set[str] = {
        'gpt-3.5-turbo', 'gpt-5-chat', 'gpt-4o-mini', 'o3-mini', 'gpt-4.1-mini', 
        'gpt-5-nano', 'gpt-5-mini', 'gpt-4.1-nano', 'o1-pro', 'o4-mini',
        'claude-sonnet-4.5', 'claude-sonnet-4', 'claude-haiku-4.5', 'claude-3-7-sonnet',
        'gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemma-3n-e4b',
        'deepseek-chat', 'deepseek-r1-0528', 'deepseek-v3.2', 'deepseek-v3', 
        'deepseek-v3.1', 'deepseek-reasoner', 'deepseek-r1',
        'qwen2.5-coder-32b', 'qwen3-omni', 'qwq-32b-fast', 'qwen3-next', 'qwen3-coder', 'qwen3-coder-big',
        'glm-4.5-air', 'glm-4.6', 'glm-4.5', 'ernie-4.5',
        'grok-code-1', 'grok-4-think', 'grok-4', 'grok-3-mini',
        'llama-4-scout', 'llama-4-maverick', 'llama-3.3',
        'mistral-small-3.1-24b', 'mistral-medium-3', 'command-a', 'sonar',
        'nemotron-ultra-235b', 'kimi-k2-0905', 'kimi-k2',
        'nova-micro', 'nova-pro', 'nova-lite', 'hermes-4-405b', 'hermes-3-405b', 
        'lucid-origin', 'goliath-120b', 'gpt-oss-120b', 'ling-1t', 'ring-1t', 'cliptagger-12b', 'seed-oss'
    }

    IMAGE_MODELS: Set[str] = {
        'dall-e-3', 'gpt-image-1', 'sd-3.5-large', 'sd-3.5', 'sdxl', 'flux-schnell', 'nano-banana'
    }

    VIDEO_MODELS: Set[str] = {
        'cogvideox-flash'
    }

    ALL_MODELS: Set[str] = CHAT_MODELS | IMAGE_MODELS | VIDEO_MODELS

    @staticmethod
    def _fetch_remote_models() -> Set[str]:
        """
        Fetches available models from G4FPro API (synchronously).
        """
        try:
            response = sync_client.get(Models.URL)

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
            response = await async_client.get(Models.URL)
            
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
        return cls._fetch_remote_models()

    @classmethod
    def get_chat_models(cls) -> Set[str]:
        """Returns all available chat-capable models."""
        return cls.CHAT_MODELS & cls._fetch_remote_models()

    @classmethod
    def get_image_models(cls) -> Set[str]:
        """Returns all available image-generation models."""
        return cls.IMAGE_MODELS & cls._fetch_remote_models()

    @classmethod
    def get_video_models(cls) -> Set[str]:
        """Returns all available video-generation models."""
        return cls.VIDEO_MODELS & cls._fetch_remote_models()

    @classmethod
    async def get_all_models_async(cls) -> Set[str]:
        """Asynchronously returns all supported models available on the server."""
        remote_models = await cls._fetch_remote_models_async()
        return remote_models

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

    @classmethod
    async def get_video_models_async(cls) -> Set[str]:
        """Asynchronously returns all available video-generation models."""
        remote_models = await cls._fetch_remote_models_async()
        return cls.VIDEO_MODELS & remote_models