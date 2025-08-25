from decoding.base_decoding import BaseStrategy
import torch
import torch.nn.functional as F
from typing import Optional

class TopKStrategy(BaseStrategy):
    """
    Top-k sampling strategy for text generation.
    - generate(): використовує HF generate з do_sample=True, top_k=...
    - custom_generate(): покроковий top-k семплінг з дебагом.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _make_generator(device, seed: Optional[int]):
        return torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        k: int = 20,
        temperature: float = 0.5,
        seed: Optional[int] = 5,
        deterministic: bool = True,
    ) -> str:
        """Generates text using built-in top-k sampling."""
        self.model.eval()

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

        if seed is not None:
            torch.manual_seed(seed)

        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out = self.model.generate(
            **enc,
            do_sample=True,                  
            top_k=k,                         
            temperature=max(temperature, 1e-8),  
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


    def custom_generate(
        self,
        prompt: str,
        debug: bool = False,
        max_new_tokens: int = 256,
        k: int = 20,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> str:
        """Deterministic top-k sampling with step-by-step debug output."""

        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

        gen = self._make_generator(self.model.device, seed)

        generated = input_ids.clone()
        eos_id = self.tokenizer.eos_token_id or getattr(self.model.config, "eos_token_id", None)
        step = 0

        if debug:
            print("\n🔎 DEBUG (top-k sampling):\n")

        with torch.no_grad():
            while True:
                outputs = self.model(input_ids=generated, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]

                # temperature
                if temperature is not None and temperature > 0.0:
                    logits = logits / temperature

                # беремо top-k
                topk_vals, topk_idx = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)

                # маскуємо все поза top-k
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(1, topk_idx, topk_vals)

                # softmax і семпл
                probs = F.softmax(masked, dim=-1)

                # torch.multinomial не приймає generator напряму на CUDA — семплимо поіменно
                if gen is not None:
                    torch.manual_seed(gen.initial_seed())
                next_token = torch.multinomial(probs, num_samples=1)

                if debug:
                    self._debug_generate(step + 1, probs, next_token.item(), top_k=min(5, k))

                # апендимо токен
                generated = torch.cat([generated, next_token], dim=-1)

                # оновлюємо attention_mask
                if attention_mask is not None:
                    pad = torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, pad], dim=-1)

                step += 1

                # стоп-умови
                if eos_id is not None and next_token.item() == eos_id:
                    if debug:
                        print("🔚 Found EOS.\n")
                    break
                if step >= max_new_tokens:
                    if debug:
                        print(f"⚠️ Limit {max_new_tokens} tokens.")
                    break

        gen_ids = generated[0, input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _debug_generate(self, step: int, probs: torch.Tensor, next_token_id: int, top_k: int = 5) -> None:
        """
        Виводить top-k ЙМОВІРНОСТЕЙ (після маскування і temperature), позначає обраний токен.
        """
        topk = torch.topk(probs, k=top_k, dim=-1)
        topk_ids = topk.indices[0].tolist()
        topk_probs = topk.values[0].tolist()

        print(f"[Step {step}] (top-k by prob)")
        for i, (token_id, p) in enumerate(zip(topk_ids, topk_probs), start=1):
            token_str = self.tokenizer.decode([token_id])
            mark = "← SELECTED" if token_id == next_token_id else ""
            print(f"  {i}. {token_str!r:20} (p={p:.4f}) {mark}")
        print()
