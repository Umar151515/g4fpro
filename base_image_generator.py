from typing import Optional, List, Dict, Any
import base64
from io import BytesIO
from pathlib import Path
from abc import ABC, abstractmethod

from PIL import Image

from .exceptions import ImageGenerationError, ImageModelNotSupportedError, ImageSaveError
from .models import Models


class BaseImageGenerator(ABC):
    """
    Abstract base class for image generation implementations.
    
    This class provides the foundation for generating images from text prompts
    and includes common functionality for model validation, payload preparation,
    and image format conversion.
    
    Attributes:
        model (Optional[str]): The image generation model to use
        n (int): Number of images to generate (1-10)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        n: int = 1
    ):
        """
        Initialize the image generator.
        
        Args:
            model: The image generation model identifier. If None, will use default
            n: Number of images to generate (default: 1, max: 10)
            
        Raises:
            ImageModelNotSupportedError: If the specified model is not supported
            TypeError: If model or n have incorrect types
            ValueError: If n is out of valid range
        """
        self.model = self._check_model(model)
        self.n = self._check_n(n)

    def _check_model(self, value: Optional[str]) -> Optional[str]:
        """
        Validate the model parameter.
        
        Args:
            value: Model identifier to validate
            
        Returns:
            Validated model string if provided and valid
            
        Raises:
            TypeError: If value is not a string
            ImageModelNotSupportedError: If model is not in supported models list
        """
        if value is None:
            return None
        
        if not isinstance(value, str):
            raise TypeError(f"'model' must be a string. Given type: {type(value).__name__}")
        elif value not in Models.IMAGE_MODELS:
            raise ImageModelNotSupportedError(value, Models.IMAGE_MODELS)
        
        return value

    def _check_n(self, value: int) -> int:
        """
        Validate the number of images parameter.
        
        Args:
            value: Number of images to generate
            
        Returns:
            Validated integer value
            
        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not between 1 and 10
        """
        if not isinstance(value, int):
            raise TypeError(f"'n' must be an integer. Given type: {type(value).__name__}")
        elif value <= 0:
            raise ValueError(f"'n' must be positive. Given: {value}")
        elif value > 10:
            raise ValueError(f"'n' cannot exceed 10. Given: {value}")
        
        return value

    def _check_prompt(self, value: str) -> str:
        """
        Validate the prompt parameter.
        
        Args:
            value: Text prompt for image generation
            
        Returns:
            Stripped and validated prompt string
            
        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty or contains only whitespace
        """
        if not isinstance(value, str):
            raise TypeError(f"'prompt' must be a string. Given type: {type(value).__name__}")
        elif not value.strip():
            raise ValueError("'prompt' cannot be empty")
        
        return value.strip()

    def _prepare_payload(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Prepare the payload for image generation API request.
        
        Args:
            prompt: Text prompt for image generation
            model: Optional model override. Uses instance model if not provided
            n: Optional number of images override. Uses instance n if not provided
            
        Returns:
            Dictionary containing the formatted payload for API request
            
        Raises:
            Various validation errors from _check_model, _check_n, and _check_prompt
        """
        prompt = self._check_prompt(prompt)
        
        final_model = model if model is not None else self.model
        if final_model is None:
            final_model = list(Models.IMAGE_MODELS)[0]
        
        final_n = n if n is not None else self.n
        
        return {
            "model": self._check_model(final_model),
            "prompt": prompt,
            "n": self._check_n(final_n)
        }

    def _image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 encoded string.
        
        Args:
            image: PIL Image object to convert
            format: Image format for encoding (default: 'PNG')
            
        Returns:
            Base64 encoded string representation of the image
            
        Raises:
            ImageGenerationError: If conversion to base64 fails
        """
        try:
            buffer = BytesIO()
            image.save(buffer, format=format)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_str
        except Exception as e:
            raise ImageGenerationError(500, f"Failed to convert image to base64: {str(e)}")

    def _save_image_to_path(self, image: Image.Image, path: str) -> str:
        """
        Save PIL Image to specified file path.
        
        Args:
            image: PIL Image object to save
            path: File path where image should be saved
            
        Returns:
            Absolute path to the saved image file
            
        Raises:
            ImageSaveError: If saving the image fails
        """
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            format = path_obj.suffix[1:].upper() if path_obj.suffix else 'PNG'
            if format not in ['PNG', 'JPEG', 'JPG', 'WEBP']:
                format = 'PNG'
                
            image.save(path_obj, format=format)
            return str(path_obj.absolute())
        except Exception as e:
            raise ImageSaveError(path, str(e))

    @property
    def url(self) -> str:
        """Get the API endpoint URL for image generation."""
        return "https://gpt4free.pro/v1/images/generations"

    @property
    def headers(self) -> Dict[str, str]:
        """Get the default headers for API requests."""
        return {"Content-Type": "application/json"}

    @abstractmethod
    def generate_urls(self, prompt: str, model: Optional[str] = None, n: Optional[int] = None) -> List[str]:
        """
        Generate image URLs from text prompt.
        
        Args:
            prompt: Text prompt for image generation
            model: Optional model override
            n: Optional number of images override
            
        Returns:
            List of URLs pointing to generated images
            
        Raises:
            ImageGenerationError: If image generation fails
            Various validation errors for prompt, model, and n parameters
        """
        pass

    @abstractmethod
    def generate_base64(self, prompt: str, model: Optional[str] = None, n: Optional[int] = None) -> List[str]:
        pass

    @abstractmethod
    def save_images(self, prompt: str, save_path: str, model: Optional[str] = None, n: Optional[int] = None) -> List[str]:
        pass

    @abstractmethod
    def _download_image(self, url: str) -> Image.Image:
        pass