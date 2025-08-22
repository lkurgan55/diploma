from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseStrategy(ABC):
    """Base class for text generation strategies."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text based on the provided prompt."""
        pass

    @abstractmethod
    def _debug_generate(self, **kwargs: dict) -> None:
        """Optional method for debugging generation step-by-step."""
        pass
