from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseStrategy(ABC):
    """Base class for text generation strategies."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        """Build-in generate text based on the provided prompt."""
        pass

    def custom_generate(self, prompt: str) -> str:
        """Cutsom generate text based on the provided prompt."""
        pass

    def _debug_generate(self, **kwargs: dict) -> None:
        """Optional method for debugging generation step-by-step."""
        pass
