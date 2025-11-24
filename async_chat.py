import json
from typing import List, Union, Optional, Dict, Any, AsyncGenerator
from httpx import TimeoutException, ConnectError

from .http_client import async_client
from .base_chat import BaseChat
from .messages import Messages
from .exceptions import (
    G4FProException,
    APIError,
    ModelNotFoundError,
    G4FProTimeoutError,
    G4FProConnectionError,
    G4FProParseError
)


class AsyncChat(BaseChat):
    """
    An asynchronous chat client for interacting with the G4FPro API.
    
    This class provides async methods to generate chat completions and streamed responses
    using various language models with configurable parameters.
    
    Attributes:
        url (str): The API endpoint URL
        headers (Dict[str, str]): HTTP headers for API requests
    """
    
    async def generate(
        self,
        message: Union[str, list, Messages],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously generate a chat completion response.
        
        Args:
            message: The input message(s) for the chat. Can be a string, list of messages,
                    or Messages object
            model: The model to use for generation. If None, uses default model
            temperature: Controls randomness (0.0 to 1.0). Lower values make output 
                        more deterministic
            top_p: Controls diversity via nucleus sampling (0.0 to 1.0). 
                   Lower values sample from more likely tokens
            max_tokens: Maximum number of tokens to generate in the response
            presence_penalty: Penalizes new tokens based on their presence in the text so far.
                            Values between -2.0 and 2.0
            frequency_penalty: Penalizes new tokens based on their frequency in the text so far.
                             Values between -2.0 and 2.0
            stop: Sequences where the API will stop generating further tokens.
                  Can be a string or list of strings
            
        Returns:
            Dict[str, Any]: The complete API response containing the generated message
            
        Raises:
            ModelNotFoundError: When the specified model is not found (404)
            APIError: When other HTTP errors occur
            G4FProTimeoutError: When the request times out
            G4FProConnectionError: When connection errors occur
            G4FProParseError: When the response cannot be parsed as JSON
            G4FProException: For other unexpected errors
            
        Example:
            >>> chat = AsyncChat()
            >>> response = await chat.generate("Hello, how are you?", model="gpt-3.5-turbo")
            >>> print(response['choices'][0]['message']['content'])
        """
        payload = self._prepare_payload(
            message=message,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            stream=False
        )
        
        try:
            response = await async_client.post(
                self.url, 
                headers=self.headers, 
                    json=payload,
            )
            
            if response.status_code != 200:
                error_message = f"HTTP error occurred: {response.status_code}"
                try:
                    error_data = response.json()
                    err = error_data.get("error")
                    error_message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                except (json.JSONDecodeError, AttributeError, ValueError):
                    error_message = response.text or error_message
                
                if response.status_code == 404:
                    raise ModelNotFoundError(response.status_code, error_message)
                else:
                    raise APIError(response.status_code, error_message)
            
            try:
                return response.json()
            except (json.JSONDecodeError, ValueError) as e:
                raise G4FProParseError(f"Failed to parse API response: {e}") from e
                
        except TimeoutException as e:
            raise G4FProTimeoutError(f"Request timed out: {e}") from e
        except ConnectError as e:
            raise G4FProConnectionError(f"Connection error: {e}") from e
        except (ModelNotFoundError, APIError, G4FProParseError):
            raise
        except Exception as e:
            raise G4FProException(f"Unexpected error during API request: {e}") from e

    async def generate_stream(
        self,
        message: Union[str, list, Messages],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a streamed chat completion response.
        
        This method yields response chunks as they become available from the API,
        providing real-time streaming of the generated content.
        
        Args:
            message: The input message(s) for the chat. Can be a string, list of messages,
                    or Messages object
            model: The model to use for generation. If None, uses default model
            temperature: Controls randomness (0.0 to 1.0). Lower values make output 
                        more deterministic
            top_p: Controls diversity via nucleus sampling (0.0 to 1.0). 
                   Lower values sample from more likely tokens
            max_tokens: Maximum number of tokens to generate in the response
            presence_penalty: Penalizes new tokens based on their presence in the text so far.
                            Values between -2.0 and 2.0
            frequency_penalty: Penalizes new tokens based on their frequency in the text so far.
                             Values between -2.0 and 2.0
            stop: Sequences where the API will stop generating further tokens.
                  Can be a string or list of strings
            
        Yields:
            str: Streamed response chunks as they become available from the API
            
        Raises:
            ModelNotFoundError: When the specified model is not found (404)
            APIError: When other HTTP errors occur
            G4FProTimeoutError: When the request times out
            G4FProConnectionError: When connection errors occur
            G4FProException: For other errors during stream generation
            
        Example:
            >>> chat = AsyncChat()
            >>> async for chunk in chat.generate_stream("Tell me a story:", model="gpt-3.5-turbo"):
            ...     print(chunk, end="", flush=True)
        """
        payload = self._prepare_payload(
            message=message,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            stream=True
        )
        
        try:
            async with async_client.stream(
                "POST",
                self.url,
                headers=self.headers,
                json=payload
            ) as response:
                
                if response.status_code != 200:
                    error_message = f"HTTP error occurred: {response.status_code}"
                    try:
                        error_data = await response.aread()
                        try:
                            error_json = json.loads(error_data)
                            err = error_json.get("error")
                            error_message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        except (json.JSONDecodeError, AttributeError, ValueError):
                            error_message = error_data.decode() if error_data else error_message
                    except Exception:
                        pass
                    
                    if response.status_code == 404:
                        raise ModelNotFoundError(response.status_code, error_message)
                    else:
                        raise APIError(response.status_code, error_message)
                
                async for line in response.aiter_lines():
                    if line.strip():
                        content = self._process_stream_chunk(line)
                        if content is not None:
                            yield content
                                
        except TimeoutException as e:
            raise G4FProTimeoutError(f"Request timed out: {e}") from e
        except ConnectError as e:
            raise G4FProConnectionError(f"Connection error: {e}") from e
        except (ModelNotFoundError, APIError):
            raise
        except Exception as e:
            raise G4FProException(f"Unexpected error during stream generation: {e}") from e