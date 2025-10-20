from typing import Union, Optional
import base64
import io

from PIL import Image

from .exceptions import ImageFormatError


class Messages:
    """
    A utility class for building and managing message lists in the format 
    required by multi-modal LLM APIs (e.g., GPT-4o, Gemini).
    """

    VALID_ROLES = {"user", "assistant", "system"}

    def __init__(
            self, 
            messages: Union[list, "Messages", None] = None
        ):
        if isinstance(messages, Messages):
            self.messages = messages.messages.copy()
        elif isinstance(messages, list):
            if not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in messages):
                raise ValueError("Invalid messages format: each message must be a dict with 'role' and 'content' keys")
            
            for msg in messages:
                if msg.get('role') not in self.VALID_ROLES:
                    raise ValueError(f"Invalid role '{msg.get('role')}'. Must be one of: {', '.join(self.VALID_ROLES)}")
                    
            self.messages = messages
        elif messages is not None:
            raise ValueError("Messages must be a list, Messages instance, or None")
        else:
            self.messages = []

    def _validate_role(self, role: str) -> None:
        """Validate that role is one of the allowed values."""
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {', '.join(self.VALID_ROLES)}")

    def add_multimodal_message(self, role: str, content: list) -> None:
        """
        Adds a message with arbitrary multi-modal content (text, image_url, etc.).
        
        Args:
            role (str): The role of the sender.
            content (List[Dict[str, Any]]): A list of content objects, 
                e.g., [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
        
        self._validate_role(role)
        
        if not content or not isinstance(content, list):
            raise ValueError("Content must be a non-empty list")
        
        if not all(isinstance(item, dict) for item in content):
            raise ValueError("All content items must be dictionaries")
            
        self.messages.append({
            "role": role,
            "content": content
        })

    def add_text_message(self, role: str, content: str) -> None:
        """
        Adds a simple text message to the list.
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
            
        self._validate_role(role)
            
        if content is None or not isinstance(content, str):
            raise ValueError("Content must be a string")
        
        self.messages.append({
            "role": role,
            "content": content
        })

    def add_url_image_message(self, role: str, image_url: str, content: Optional[str] = None) -> None:
        """
        Adds a multi-modal message by directly referencing an image URL (can be a public URL 
        or a Base64 Data URI). This is the primary method for image inclusion.
        
        Args:
            role (str): The role of the sender.
            image_url (str): The public URL or a Base64 Data URI of the image.
            content (Optional[str]): Optional text description for the image.
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
            
        self._validate_role(role)
            
        if not image_url or not isinstance(image_url, str):
            raise ValueError("Image URL must be a non-empty string")
            
        if content is not None and not isinstance(content, str):
            raise ValueError("Content must be a string or None")
        
        content_list = []
        if content:
             content_list.append({"type": "text", "text": content})

        content_list.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
        
        self.add_multimodal_message(role, content_list)

    def add_base64_image_message(self, role: str, image: str, content: Optional[str] = None) -> None:
        """
        Adds a multi-modal message containing a Base64 encoded image.
        
        This method ensures the input Base64 string is correctly formatted as a Data URI 
        before passing it to add_url_image_message.
        
        Args:
            role (str): The role of the sender.
            image (str): The Base64 string of the image (either pure data or full Data URI).
            content (Optional[str]): Optional text description for the image.
            
        Raises:
            ImageFormatError: If the image format cannot be determined from the raw Base64 data.
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
            
        self._validate_role(role)
            
        if not image or not isinstance(image, str):
            raise ValueError("Image must be a non-empty string")
            
        if content is not None and not isinstance(content, str):
            raise ValueError("Content must be a string or None")
        
        mime_type = self._extract_mime_type(image)

        if mime_type is None:
            guessed_mime = self._determine_mime_with_pillow(image)
            
            if guessed_mime:
                full_image_url = f"data:{guessed_mime};base64,{image}"
            else:
                raise ImageFormatError("Failed to determine image format from pure Base64 data via Pillow.")

        else:
            full_image_url = image

        self.add_url_image_message(role, full_image_url, content)

    def add_file_image_message(self, role: str, file_path: str, content: Optional[str] = None) -> None:
        """
        Adds a multi-modal message by reading an image from a local file path 
        and encoding it as a Base64 Data URI.
        
        Args:
            role (str): The role of the sender.
            file_path (str): The local path to the image file.
            content (Optional[str]): Optional text description for the image.
            
        Raises:
            ImageFormatError: If the file is not found or is not a valid image.
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
            
        self._validate_role(role)
            
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
            
        if content is not None and not isinstance(content, str):
            raise ValueError("Content must be a string or None")
        
        full_image_url = self.file_to_base64_uri(file_path)

        self.add_url_image_message(role, full_image_url, content)

    def _extract_mime_type(self, base64_uri: str) -> Optional[str]:
        if not base64_uri or not isinstance(base64_uri, str):
            return None
            
        if base64_uri.startswith('data:'):
            try:
                mime_part = base64_uri.split(';base64,', 1)[0]
                mime_type = mime_part.split('data:', 1)[1]
                return mime_type
            except (IndexError, AttributeError):
                return None
        return None

    def _determine_mime_with_pillow(self, base64_content: str) -> Optional[str]:
        if not base64_content or not isinstance(base64_content, str):
            return None
            
        try:
            image_bytes = base64.b64decode(base64_content)
        except (base64.binascii.Error, ValueError, TypeError):
            return None

        if not image_bytes:
            return None
            
        image_stream = io.BytesIO(image_bytes)

        try:
            image = Image.open(image_stream)
            
            if image.format:
                return f"image/{image.format.lower()}"
            
            return None
            
        except (IOError, OSError, Exception):
            return None

    @staticmethod
    def file_to_base64_uri(file_path: str) -> str:
        """
        Reads an image from a local file path, determines its format, and encodes 
        it into a full Base64 Data URI string ('data:image/jpeg;base64,...').
        
        Args:
            file_path (str): The local file path to the image.
            
        Returns:
            str: The full Base64 Data URI string.
            
        Raises:
            ImageFormatError: If the file is not a valid image format.
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
            
        try:
            with open(file_path, "rb") as image_file:
                image_bytes = image_file.read()
        except FileNotFoundError:
            raise ImageFormatError(f"Image file not found at path: {file_path}")
        except (IOError, OSError, PermissionError) as e:
            raise ImageFormatError(f"Error reading file at {file_path}: {e}")

        if not image_bytes:
            raise ImageFormatError(f"File at {file_path} is empty")

        image_stream = io.BytesIO(image_bytes)
        try:
            image = Image.open(image_stream)
            img_format = image.format
        except (IOError, OSError, Exception) as e:
            raise ImageFormatError(f"File at {file_path} is not a valid image: {e}")

        if not img_format:
            raise ImageFormatError(f"Could not determine format of image file at: {file_path}")

        try:
            encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        except (TypeError, ValueError) as e:
            raise ImageFormatError(f"Error encoding image to base64: {e}")

        mime_type = f"image/{img_format.lower()}"
        return f"data:{mime_type};base64,{encoded_string}"