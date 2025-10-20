from typing import Optional, List, Union, Dict, Any
import json
from abc import ABC, abstractmethod

from .exceptions import (
    ModelNotSupportedError, InvalidMessageFormatError, G4FProParseError
)
from .models import Models
from .messages import Messages


class BaseChat(ABC):
    def __init__(
        self, 
        model: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ):
        self.model = self._check_model(model)
        self.temperature = self._check_temperature(temperature)
        self.top_p = self._check_top_p(top_p)
        self.max_tokens = self._check_max_tokens(max_tokens)
        self.presence_penalty = self._check_penalty(presence_penalty, "presence_penalty")
        self.frequency_penalty = self._check_penalty(frequency_penalty, "frequency_penalty")
        self.stop = self._check_stop(stop)

    def _check_model(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        
        if not isinstance(value, str):
            raise TypeError(f"'model' must be a string. Given type: {type(value).__name__}")
        elif value not in Models.CHAT_MODELS:
            raise ModelNotSupportedError(f"'model' doesn't exist. Given: {value}. Available models: {Models.CHAT_MODELS}")
        
        return value
    
    def _check_temperature(self, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"'temperature' must be a float or integer. Given type: {type(value).__name__}")
        elif not (0.0 <= value <= 2.0):
            raise ValueError(f"'temperature' must be between 0.0 and 2.0. Given: {value}")
        
        return value

    def _check_top_p(self, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"'top_p' must be a float or integer. Given type: {type(value).__name__}")
        elif not (0.0 <= value <= 1.0):
            raise ValueError(f"'top_p' must be between 0.0 and 1.0. Given: {value}")
        
        return value

    def _check_max_tokens(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        elif not isinstance(value, int) or value <= 0:
            raise ValueError(f"'max_tokens' must be a positive integer or None. Given type: {type(value).__name__}")
        
        return value

    def _check_penalty(self, value: float, name: str) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Penalty '{name}' must be a float or integer. Given type: {type(value).__name__}")
        elif not (-2.0 <= value <= 2.0):
            raise ValueError(f"Penalty '{name}' must be between -2.0 and 2.0. Given: {value}")
        
        return value

    def _check_stop(self, value: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        if value is None:
            return None
        elif isinstance(value, str):
            value = [value]
        elif not isinstance(value, list) or any(not isinstance(s, str) for s in value):
            raise TypeError(f"'stop' must be a string or a list of strings. Given type: {type(value).__name__}")
        elif len(value) > 4:
            raise ValueError(f"'stop' can contain a maximum of 4 elements. Given: {len(value)}")
            
        return value

    def _prepare_payload(
        self,
        message: Union[str, list, Messages],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        if not isinstance(message, (str, list, Messages)):
            raise InvalidMessageFormatError(
                f"Invalid 'message' format. Expected str, list or Messages. Got: {type(message).__name__}"
            )
        
        if isinstance(message, str):
            messages_payload = [{"role": "user", "content": message}]
        elif isinstance(message, list):
            messages_payload = message
        elif isinstance(message, Messages):
            messages_payload = message.messages
        
        final_model = model if model is not None else self.model
        if final_model is None:
            final_model = list(Models.CHAT_MODELS)[0]
        
        params = {
            "model": self._check_model(final_model),
            "messages": messages_payload,
            "temperature": self._check_temperature(temperature) if temperature is not None else self.temperature,
            "top_p": self._check_top_p(top_p) if top_p is not None else self.top_p,
            "max_tokens": self._check_max_tokens(max_tokens) if max_tokens is not None else self.max_tokens,
            "presence_penalty": self._check_penalty(presence_penalty, "presence_penalty") if presence_penalty is not None else self.presence_penalty,
            "frequency_penalty": self._check_penalty(frequency_penalty, "frequency_penalty") if frequency_penalty is not None else self.frequency_penalty,
            "stop": self._check_stop(stop) if stop is not None else self.stop
        }
        
        if stream:
            params["stream"] = True
        
        return {
            k: v for k, v in params.items() if v is not None
        }

    def _process_stream_chunk(self, line: str) -> Optional[str]:
        decoded_line = line.decode('utf-8').strip() if hasattr(line, 'decode') else line.strip()
        
        if not decoded_line:
            return None
            
        if not decoded_line.startswith('data:'):
            return None
            
        data = decoded_line[len('data: '):]
        
        if data.strip() == '[DONE]':
            return None
        
        try:
            chunk = json.loads(data)
            if not chunk.get("choices"):
                return None
            content = chunk["choices"][0].get("delta", {}).get("content")
            return content
        except json.JSONDecodeError as e:
            raise G4FProParseError(f"Failed to parse streaming chunk: {e}") from e
        except KeyError:
            return None

    @property
    def url(self) -> str:
        return "https://gpt4free.pro/v1/chat/completions"

    @property
    def headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    @abstractmethod
    def generate(
        self,
        message: Union[str, list, Messages],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None
    ):
        pass

    @abstractmethod
    def generate_stream(
        self,
        message: Union[str, list, Messages],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None
    ):
        pass