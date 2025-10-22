from typing import Optional, List
import json
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .exceptions import ImageGenerationError
from .base_image_generator import BaseImageGenerator


class AsyncImageGenerator(BaseImageGenerator):
    """
    Asynchronous image generator for interacting with G4FPro API.
    
    This class provides asynchronous methods for generating images from text descriptions
    with the ability to get results as URLs, base64 strings, or save to files.
    
    Attributes:
        url (str): API endpoint URL
        headers (Dict[str, str]): HTTP headers for API requests
    """

    async def generate_urls(self, prompt: str, model: Optional[str] = None, n: Optional[int] = None) -> List[str]:
        """
        Generate images and return a list of URLs.
        
        Args:
            prompt: Text description of the image to generate
            model: Model to use for generation. If None, uses default model
            n: Number of images to generate. If None, uses default value
            
        Returns:
            List[str]: List of URLs of generated images
            
        Raises:
            ImageModelNotSupportedError: When specified model is not found
            ImageGenerationError: When image generation errors occur
        """
        payload = self._prepare_payload(prompt, model, n)
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(self.url, headers=self.headers, json=payload)
                
                if response.status_code != 200:
                    error_message = f"HTTP error occurred: {response.status_code}"
                    try:
                        error_data = response.json()
                        err = error_data.get("error")
                        error_message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    except (json.JSONDecodeError, AttributeError, ValueError):
                        error_message = response.text or error_message
                    
                    raise ImageGenerationError(response.status_code, error_message)
                
                data = response.json()
                
                if data and data.get("data"):
                    urls = [item.get("url") for item in data["data"] if item.get("url")]
                    if urls:
                        return urls
                    else:
                        raise ImageGenerationError(500, "API returned empty URLs in response")
                else:
                    raise ImageGenerationError(500, "API returned empty data in response")
                    
        except httpx.TimeoutException as e:
            raise ImageGenerationError(408, f"Request timeout: {e}")
        except httpx.ConnectError as e:
            raise ImageGenerationError(503, f"Connection error: {e}")
        except ImageGenerationError:
            raise
        except Exception as e:
            raise ImageGenerationError(500, f"Unexpected error: {e}")

    async def generate_base64(self, prompt: str, model: Optional[str] = None, n: Optional[int] = None) -> List[str]:
        """
        Generate images and return them as base64 strings.
        
        Args:
            prompt: Text description of the image to generate
            model: Model to use for generation. If None, uses default model
            n: Number of images to generate. If None, uses default value
            
        Returns:
            List[str]: List of base64 strings of generated images
            
        Raises:
            ImageModelNotSupportedError: When specified model is not found
            ImageGenerationError: When image generation errors occur
        """
        urls = await self.generate_urls(prompt, model, n)
        base64_images = []
        
        for url in urls:
            image = await self._download_image(url)
            base64_str = self._image_to_base64(image)
            base64_images.append(base64_str)
            
        return base64_images

    async def save_images(self, prompt: str, save_path: str = "generated_image", model: Optional[str] = None, n: Optional[int] = None) -> List[str]:
        """
        Generate and save images to the specified path.
        
        Args:
            prompt: Text description of the image to generate
            save_path: Path for saving. Can be:
                    1. Full file path (e.g., 'folder/cat.jpg'). 
                        If n > 1, becomes 'folder/cat_1.jpg', 'cat_2.jpg', etc.
                    2. Directory path (e.g., 'folder/').
                        File name will be generated (e.g., 'folder/img_1.png').
                    3. Base name (e.g., 'cat'). 
                        Saves as 'cat.png' or 'cat_1.png' in current directory.
            model: Model to use for generation.
            n: Number of images to generate. 
            
        Returns:
            List[str]: List of absolute paths to saved files
            
        Raises:
            ImageModelNotSupportedError: When specified model is not found
            ImageGenerationError: When image generation errors occur
            ImageSaveError: When image saving fails
        """
        urls = await self.generate_urls(prompt, model, n)
        saved_paths = []
        
        path_obj = Path(save_path)
        
        default_suffix = ".png"
        
        base_name = ""
        save_directory = Path(".")

        if path_obj.suffix and path_obj.name not in ('.', '..'):
            base_name = path_obj.stem
            save_directory = path_obj.parent
            default_suffix = path_obj.suffix
        else:
            if path_obj.is_dir() or str(path_obj).endswith(('\\', '/')):
                save_directory = path_obj
                base_name = "img" 
            else:
                base_name = path_obj.name
                
        for i, url in enumerate(urls):
            image = await self._download_image(url)
            
            if len(urls) > 1:
                final_file_name = f"{base_name}_{i+1}{default_suffix}"
            else:
                final_file_name = f"{base_name}{default_suffix}"
                
            final_path = str(save_directory / final_file_name)
            
            if not save_directory.exists() and save_directory != Path('.'):
                save_directory.mkdir(parents=True, exist_ok=True)
                
            saved_path = self._save_image_to_path(image, final_path)
            saved_paths.append(saved_path)
                
        return saved_paths

    async def _download_image(self, url: str) -> Image.Image:
        """
        Download image from URL and return PIL Image object.
        
        Args:
            url: URL of the image to download
            
        Returns:
            Image.Image: PIL Image object
            
        Raises:
            ImageGenerationError: When image download fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                
                image_data = BytesIO(response.content)
                return Image.open(image_data)
        except Exception as e:
            raise ImageGenerationError(500, f"Failed to download image from {url}: {str(e)}")