# G4FPro - Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Models](#models)
5. [Working with Chat](#working-with-chat)
6. [Image Generation](#image-generation)
7. [Multimodal Capabilities](#multimodal-capabilities)
8. [Exceptions and Error Handling](#exceptions-and-error-handling)
9. [Limitations](#limitations)
10. [Notes and Known Issues](#notes-and-known-issues)

## Introduction

G4FPro is a Python wrapper library for the gpt4free.pro service that provides free access to various LLM models without the need for API keys. The library supports both synchronous and asynchronous requests.

**Main Features:**
- Text generation with streaming support
- Image generation from text descriptions
- Multimodal requests (text + images)
- Support for multiple models (GPT, Claude, Gemini, DeepSeek, etc.)
- Synchronous and asynchronous API

## Installation

```bash
pip install g4fpro
```

## Quick Start

### Simple Text Request

```python
from g4fpro import Chat

# Create chat client
chat = Chat()

# Simple request
response = chat.generate("Hello! How are you?")
print(response['choices'][0]['message']['content'])
```

### Image Generation

```python
from g4fpro import ImageGenerator

# Create image generator
generator = ImageGenerator()

# Generate and save image
saved_paths = generator.save_images("beautiful cat in the garden", "cat.png")
print(f"Image saved: {saved_paths[0]}")
```

## Models

### Getting List of Available Models

```python
from g4fpro import Models

# Synchronously
all_models = Models.get_all_models()
chat_models = Models.get_chat_models()
image_models = Models.get_image_models()

print(f"Available chat models: {chat_models}")
print(f"Available image generation models: {image_models}")

# Asynchronously
import asyncio

async def get_models_async():
    all_models = await Models.get_all_models_async()
    return all_models

models = asyncio.run(get_models_async())
```

### Main Model Categories

**Chat Models:**
- `gpt-3.5-turbo`, `gpt-5-chat`, `gpt-4o-mini`, `o3-mini`
- `claude-sonnet-4.5`, `claude-haiku-4.5`
- `gemini-2.5-flash-lite`, `gemini-2.5-pro`
- `deepseek-chat`, `deepseek-reasoner`
- And many more...

**Image Generation Models:**
- `dall-e-3`, `gpt-image-1`
- `sd-3.5-large`, `sdxl`
- `flux-schnell`, `nano-banana`

## Working with Chat

### Synchronous Client

```python
from g4fpro import Chat

# Creating client with default settings
chat = Chat()

# Or with custom settings
chat = Chat(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000
)

# Simple request
response = chat.generate("Write a short story about space")
print(response['choices'][0]['message']['content'])

# Request with additional parameters
response = chat.generate(
    message="Explain the theory of relativity",
    model="claude-sonnet-4.5",
    temperature=0.3,
    max_tokens=500
)
```

### Stream Generation

```python
from g4fpro import Chat

chat = Chat()

# Stream response generation
for chunk in chat.generate_stream("Tell a story about a dragon"):
    print(chunk, end="", flush=True)
```

### Asynchronous Client

```python
from g4fpro import AsyncChat
import asyncio

async def main():
    chat = AsyncChat()
    
    # Regular request
    response = await chat.generate("Hello! How are you?")
    print(response['choices'][0]['message']['content'])
    
    # Stream request
    async for chunk in chat.generate_stream("Write a poem about the sea"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Working with Message History

```python
from g4fpro import Chat, Messages

# Create message history
messages = Messages()
messages.add_text_message("system", "You are a helpful assistant that responds in Russian")
messages.add_text_message("user", "Hello! What's your name?")
messages.add_text_message("assistant", "Hello! I am a virtual assistant created to help users. How can I help you?")
messages.add_text_message("user", "Tell me about yourself")

chat = Chat()
response = chat.generate(messages)
print(response['choices'][0]['message']['content'])
```

## Image Generation

### Synchronous Generation

```python
from g4fpro import ImageGenerator

generator = ImageGenerator()

# Getting image URLs
urls = generator.generate_urls("futuristic city at night with neon lights", model="dall-e-3")
print(f"Generated URLs: {urls}")

# Getting base64
base64_images = generator.generate_base64("sunset in the mountains", n=2)
print(f"Received {len(base64_images)} images in base64")

# Saving images
saved_paths = generator.save_images(
    "cute kitten playing with a ball of yarn",
    "kitten.jpg",
    n=1
)
print(f"Images saved: {saved_paths}")
```

### Asynchronous Generation

```python
from g4fpro import AsyncImageGenerator
import asyncio

async def main():
    generator = AsyncImageGenerator()
    
    # URL generation
    urls = await generator.generate_urls("spaceship in a distant galaxy")
    print(f"URLs: {urls}")
    
    # Saving images
    paths = await generator.save_images("abstract art", "artwork.png")
    print(f"Saved: {paths}")

asyncio.run(main())
```

### Image Saving Options

```python
from g4fpro import ImageGenerator

generator = ImageGenerator()

# 1. Full path with suffix
paths = generator.save_images("landscape", "images/landscape.jpg")

# 2. Directory only
paths = generator.save_images("portrait", "my_images/")

# 3. Base name
paths = generator.save_images("still life", "still_life")

# 4. Multiple images
paths = generator.save_images("different dogs", "dogs", n=3)
# Will save: dogs_1.png, dogs_2.png, dogs_3.png
```

## Multimodal Capabilities

### Working with Images and Text

```python
from g4fpro import Messages, Chat

messages = Messages()

# Text message
messages.add_text_message("user", "Look at this image and describe what you see")

# Adding image by URL
messages.add_url_image_message(
    "user", 
    "https://example.com/image.jpg",
    "Here is an image for analysis"
)

# Adding local image
messages.add_file_image_message(
    "user",
    "path/to/local/image.png",
    "Analyze this local image"
)

# Adding base64 image
with open("image.jpg", "rb") as f:
    import base64
    base64_data = base64.b64encode(f.read()).decode('utf-8')
    messages.add_base64_image_message("user", base64_data, "What is in this image?")

chat = Chat()
response = chat.generate(messages, model="gpt-4o-mini")
print(response['choices'][0]['message']['content'])
```

### Complex Multimodal Requests

```python
from g4fpro import Messages

messages = Messages()

# Message with multiple content types
content = [
    {"type": "text", "text": "Analyze this image and answer the questions:"},
    {"type": "text", "text": "1. What is shown in the picture?"},
    {"type": "text", "text": "2. What colors predominate?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/art.jpg"}}
]

messages.add_multimodal_message("user", content)
```

## Exceptions and Error Handling

### Main Exceptions

```python
from g4fpro.exceptions import (
    G4FProException,
    APIError,
    ModelNotFoundError,
    G4FProTimeoutError,
    G4FProConnectionError,
    ImageGenerationError
)

try:
    chat = Chat()
    response = chat.generate("Hello", model="non-existent-model")
    
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
    
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
    
except G4FProTimeoutError as e:
    print(f"Request timeout: {e}")
    
except G4FProConnectionError as e:
    print(f"Connection issues: {e}")
    
except ImageGenerationError as e:
    print(f"Image generation error: {e}")
```

## Limitations

### Usage Limits

- **Text generation**: ~5 requests per minute
- **Image generation**: ~1 request per 30 seconds
- **Number of images**: up to 10 per one request

## Notes and Known Issues

### Implementation Features

1. **Service instability**: The gpt4free.pro service may be unavailable at any time without warning

2. **Multimodal capabilities**: All chat models, including even `gpt-3.5-turbo`, support working with images. Probably, when sending images, automatic model substitution occurs to a more advanced one (for example, `gpt-5-nano`)

3. **Code quality**: The library is written by a not very experienced developer, there may be shortcomings and non-optimal solutions

4. **API documentation**: The official API documentation is minimal, many things are implemented by trial and error

### Comprehensive Usage Example

```python
from g4fpro import Chat, ImageGenerator, Messages
from g4fpro.exceptions import G4FProException
import time

class G4FProClient:
    def __init__(self):
        self.chat = Chat(model="gpt-4o-mini")
        self.image_generator = ImageGenerator(model="sd-3.5-large")
        
    def analyze_image_with_text(self, image_base64, question):
        """Analyzes an image and answers a question about it"""
        messages = Messages()
        messages.add_base64_image_message("user", image_base64, question)
        
        try:
            response = self.chat.generate(messages)
            return response['choices'][0]['message']['content']
        except G4FProException as e:
            return f"Error analyzing image: {e}"
    
    def generate_and_analyze(self, prompt):
        """Generates an image and analyzes it"""
        try:
            # Generate image
            image_base64 = self.image_generator.generate_base64(prompt)[0]
            time.sleep(35)  # Wait before next request
            
            # Analyze generated image
            analysis = self.analyze_image_with_text(image_base64, "Describe what is shown in this picture")
            
            return analysis
            
        except G4FProException as e:
            return f"Error: {e}"

# Usage
client = G4FProClient()
result = client.generate_and_analyze("fantastic landscape with two suns")
print(result)
```