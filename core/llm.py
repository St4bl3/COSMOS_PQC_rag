# --- File: core/llm.py ---
import asyncio
from groq import Groq, AsyncGroq
from typing import Dict, Any, Optional, List # Added List
import config
import logging # Use logging

# --- LLM Interaction Wrapper (using GROQ) ---

class LLMClient:
    """
    Wrapper for interacting with the LLM via the GROQ API.
    Handles synchronous and asynchronous calls.
    """
    def __init__(self, api_key: str = config.GROQ_API_KEY, model: str = config.GROQ_LLM_MODEL):
        if not api_key:
            raise ValueError("GROQ API key is required.")
        self.api_key = api_key
        self.model = model
        # Initialize clients (consider lazy initialization if preferred)
        try:
            self.sync_client = Groq(api_key=self.api_key)
            self.async_client = AsyncGroq(api_key=self.api_key)
            logging.info(f"LLM Client initialized for model: {self.model}")
        except Exception as e:
            logging.error(f"Failed to initialize Groq clients: {e}")
            raise

    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Synchronously generate text using the specified model."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            logging.debug(f"Sending prompt to GROQ model {self.model}: {prompt[:100]}...")
            chat_completion = self.sync_client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=kwargs.get("temperature", 0.7), # Default temperature
                max_tokens=kwargs.get("max_tokens", 1024), # Default max tokens
                # Add other parameters like top_p, stop sequences as needed
            )
            response_content = chat_completion.choices[0].message.content
            logging.debug(f"Received response from GROQ: {response_content[:100]}...")
            return response_content
        except Exception as e:
            logging.error(f"Error during GROQ API call: {e}")
            # Implement more robust error handling (e.g., retries, logging)
            raise

    async def agenerate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Asynchronously generate text using the specified model."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            logging.debug(f"Sending async prompt to GROQ model {self.model}: {prompt[:100]}...")
            chat_completion = await self.async_client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            response_content = chat_completion.choices[0].message.content
            logging.debug(f"Received async response from GROQ: {response_content[:100]}...")
            return response_content
        except Exception as e:
            logging.error(f"Error during async GROQ API call: {e}")
            # Implement more robust error handling
            raise
