from typing import Union, List, Dict, Any, Optional
from openai import AzureOpenAI
from time import sleep
from os import environ


class GPT4API:
    """
    A wrapper class for interacting with OpenAI's GPT-4 API through Azure.

    This class provides methods to send prompts to GPT-4 models and retrieve responses.
    It handles authentication, request formatting, and error handling.
    """

    def __init__(
        self, model_name: str = "gpt-4o-mini", temperature: float = 0.0
    ) -> None:
        """
        Initialize the GPT4API with model settings and API credentials.

        Args:
            model_name: The name of the GPT-4 model to use. Must be one of ['gpt-4o', 'gpt-4o-mini'].
            temperature: Controls randomness in the model's output. Higher values (e.g., 0.8) make output
                         more random, while lower values (e.g., 0.2) make it more deterministic.

        Note:
            TODO: gpt-4 is no longer reproducible. Should switch to open source models.
        """
        endpoint = environ["AZURE_OPENAI_ENDPOINT"]
        api_key = environ["AZURE_OPENAI_API_KEY"]
        assert isinstance(model_name, str)
        assert model_name in ["gpt-4o", "gpt-4o-mini"]
        assert isinstance(temperature, float)
        self.model_name = model_name

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2023-12-01-preview",
        )

        self.temperature = temperature
        self.prompt = "{question}"

    def get_response(self, question: Union[str, List[Dict[str, str]]]) -> str:
        """
        Send a question to the GPT-4 model and get a response.

        Args:
            question: Either a string containing the question or a list of message dictionaries
                     where each dictionary has 'role' and 'content' keys.

        Returns:
            The model's response as a string.

        Raises:
            Exception: If there's an error communicating with the API.
        """
        if isinstance(question, str):
            question = question
        elif isinstance(question, list):
            question = question[0]["content"]

        assert isinstance(question, str)
        instruction = self.prompt.format(question=question)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": instruction}],
                temperature=self.temperature,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except Exception as e:
            print(e)
            return f"Error: {str(e)}"

        response_text = response.choices[0].message.content
        assert isinstance(response_text, str)

        sleep(0.5)  # Rate limiting to avoid hitting API limits
        return response_text

    def set_model(self, model_name: str) -> None:
        """
        Change the model being used.

        Args:
            model_name: The name of the new model to use.
        """
        self.model_name = model_name

    def set_prompt(self, prompt: str) -> None:
        """
        Set a new prompt template.

        Args:
            prompt: The new prompt template string. Should contain {question} placeholder.
        """
        self.prompt = prompt

    def set_temperature(self, temperature: float) -> None:
        """
        Set a new temperature value for the model.

        Args:
            temperature: The new temperature value (typically between 0.0 and 1.0).
        """
        self.temperature = temperature
