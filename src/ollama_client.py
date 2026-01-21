"""Ollama API client for analyzing radiology reports."""

import json
import logging
import time
import requests
from typing import Dict, Optional
import config
import prompts


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = config.OLLAMA_BASE_URL,
                 model: str = config.MODEL_NAME):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
        """
        self.base_url = base_url
        self.model = model
        self.chat_url = f"{base_url}/api/chat"
        self.tags_url = f"{base_url}/api/tags"

    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available.

        Returns:
            True if Ollama is running and model is available
        """
        try:
            response = requests.get(self.tags_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(self.model in m.get('name', '') for m in models)
            return False
        except Exception as e:
            logging.error(f"Failed to check Ollama availability: {e}")
            return False

    def _build_analysis_prompt(self, report_text: str) -> str:
        """Build the analysis prompt for the model.

        Args:
            report_text: Full radiology report text

        Returns:
            Formatted prompt string
        """
        return prompts.build_analysis_prompt(report_text)

    def _query_ollama(self, prompt: str, retry_count: int = 0) -> Optional[Dict[str, str]]:
        """Send query to Ollama API with retry logic.

        Args:
            prompt: The prompt to send
            retry_count: Current retry attempt

        Returns:
            Dict with 'content' and 'thinking' keys, or None if failed
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": "json",  # Force JSON output
            "think": config.REASONING_EFFORT,  # GPT-OSS reasoning effort: "low", "medium", "high"
            "options": {
                "temperature": config.TEMPERATURE,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=config.TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            message = result.get('message', {})
            return {
                'content': message.get('content', ''),
                'thinking': message.get('thinking', '')
            }

        except requests.Timeout:
            if retry_count < config.MAX_RETRIES:
                wait_time = 2 ** retry_count  # Exponential backoff
                logging.warning(f"Timeout, retrying in {wait_time}s... (attempt {retry_count + 1}/{config.MAX_RETRIES})")
                time.sleep(wait_time)
                return self._query_ollama(prompt, retry_count + 1)
            else:
                logging.error(f"Request timed out after {config.MAX_RETRIES} retries")
                return None

        except requests.RequestException as e:
            if retry_count < config.MAX_RETRIES:
                wait_time = 2 ** retry_count
                logging.warning(f"Request error: {e}, retrying in {wait_time}s... (attempt {retry_count + 1}/{config.MAX_RETRIES})")
                time.sleep(wait_time)
                return self._query_ollama(prompt, retry_count + 1)
            else:
                logging.error(f"Ollama API error after {config.MAX_RETRIES} retries: {e}")
                return None

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse JSON response from the model.

        Args:
            response: Raw response string from model

        Returns:
            Dict with 'is_imaging_related' and 'addendum_content' keys
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)

            # Validate required keys
            if 'is_imaging_related' not in data or 'addendum_content' not in data:
                logging.error(f"Missing required keys in response: {response[:200]}")
                return {"is_imaging_related": "Error", "addendum_content": "Missing keys in response"}

            # Normalize the is_imaging_related value
            is_related = str(data['is_imaging_related']).strip()
            if is_related.lower() in ['yes', 'true']:
                is_related = 'Yes'
            elif is_related.lower() in ['no', 'false']:
                is_related = 'No'

            return {
                "is_imaging_related": is_related,
                "addendum_content": str(data['addendum_content'])
            }

        except json.JSONDecodeError:
            # Fallback: try to extract Yes/No from text
            logging.warning(f"Failed to parse JSON response, attempting text extraction: {response[:200]}")

            response_lower = response.lower()
            if 'yes' in response_lower or 'true' in response_lower:
                return {"is_imaging_related": "Yes", "addendum_content": response}
            elif 'no' in response_lower or 'false' in response_lower:
                return {"is_imaging_related": "No", "addendum_content": "None"}
            else:
                return {"is_imaging_related": "Error", "addendum_content": f"Failed to parse: {response[:200]}"}

    def _has_addendum_marker(self, report_text: str) -> bool:
        """Check if report contains addendum markers.

        Args:
            report_text: Full radiology report text

        Returns:
            True if addendum marker found, False otherwise
        """
        if not report_text:
            return False

        # Convert to lowercase for case-insensitive search
        text_lower = report_text.lower()

        # Check for various addendum markers
        addendum_markers = [
            'addendum',
            'addenda',
            '** addendum',
            '******** addendum'
        ]

        return any(marker in text_lower for marker in addendum_markers)

    def analyze_report(self, report_text: str) -> Dict[str, str]:
        """Analyze a radiology report for imaging-related addenda.

        Args:
            report_text: Full radiology report text

        Returns:
            Dict with 'is_imaging_related' ("Yes"/"No"/"Error"),
            'addendum_content' (extracted text or "None"),
            and 'reasoning' (model's thinking process)
        """
        # Handle empty report
        if not report_text or not report_text.strip():
            logging.warning("Empty report text provided")
            return {"is_imaging_related": "No", "addendum_content": "None", "reasoning": "Empty report"}

        # Pre-check: Skip model analysis if no addendum marker present
        if not self._has_addendum_marker(report_text):
            logging.info("No addendum marker found in report, skipping model analysis")
            return {"is_imaging_related": "No", "addendum_content": "None", "reasoning": "No addendum marker found"}

        # Build prompt
        prompt = self._build_analysis_prompt(report_text)

        # Query Ollama
        response_dict = self._query_ollama(prompt)

        # Handle query failure
        if response_dict is None:
            return {"is_imaging_related": "Error", "addendum_content": "Ollama query failed", "reasoning": "Query failed"}

        # Extract content and thinking
        content = response_dict.get('content', '')
        thinking = response_dict.get('thinking', '')

        # Parse response
        result = self._parse_response(content)
        result['reasoning'] = thinking

        return result
