from typing import List


class G4FProException(Exception): 
    pass

class APIError(G4FProException):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error ({status_code}): {message}")

class ModelNotFoundError(APIError):
    pass

class G4FProTimeoutError(G4FProException):
    pass

class G4FProConnectionError(G4FProException):
    pass

class G4FProParseError(G4FProException):
    pass

class ImageFormatError(G4FProException):
    pass

class InvalidMessageFormatError(Exception):
    pass

class ModelNotSupportedError(G4FProException):
    pass

# Кастомные ошибки для генерации изображений
class ImageModelNotSupportedError(Exception):
    """Исключение, когда модель изображений не поддерживается"""
    def __init__(self, model: str, available_models: List[str]):
        self.model = model
        self.available_models = available_models
        super().__init__(f"Image model '{model}' is not supported. Available models: {available_models}")

class ImageGenerationError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Image generation failed (Status {status_code}): {message}")

class ImageSaveError(Exception):
    def __init__(self, path: str, error: str):
        self.path = path
        self.error = error
        super().__init__(f"Failed to save image to {path}: {error}")

class InvalidImageFormatError(Exception):
    def __init__(self, format: str):
        self.format = format
        super().__init__(f"Invalid image format: {format}. Supported formats: url, base64, save")