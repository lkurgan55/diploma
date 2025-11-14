from decoding.base_decoding import BaseStrategy
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "5"

import torch
import torch.nn.functional as F
from typing import Optional

class TopPStrategy(BaseStrategy):
    """Top-p (nucleus) sampling strategy for text generation."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        p: float = 0.9,
        temperature: float = 0.5,
    ) -> str:
        """Generates text using built-in top-p (nucleus) sampling."""
        self.model.eval()

        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out = self.model.generate(
            **enc,
            do_sample=True,
            top_p=p,
            temperature=temperature,
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
        p: float = 0.9,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        deterministic: bool = False,
        min_tokens_to_keep: int = 1,
    ) -> str:
        """Top-p (nucleus) sampling with step-by-step debug output."""
        self.model.eval()

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

        gen = self._make_generator(self.model.device, seed)

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        generated = input_ids.clone()
        eos_id = self.tokenizer.eos_token_id or getattr(self.model.config, "eos_token_id", None)
        step = 0
        p = float(max(min(p, 1.0), 1e-8))  # clamp p

        if debug:
            print("\n🔎 DEBUG (top-p sampling):\n")

        with torch.no_grad():
            while True:
                outputs = self.model(input_ids=generated, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]

                # temperature
                t = max(temperature, 1e-8)
                logits = logits / t

                # ---- nucleus filtering (in-place на копії logits) ----
                # сортуємо логіти за спаданням
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # маска: усе, що поза порогом p, видаляємо
                sorted_indices_to_remove = cumulative_probs > p

                # зрушення, щоб залишити токен на межі (включно)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # гарантуємо щонайменше min_tokens_to_keep
                if min_tokens_to_keep > 1:
                    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

                # переносимо маску назад у несортований простір індексів
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # працюємо на копії, щоб не псувати вихідні логіти
                filtered_logits = logits.clone()
                filtered_logits.scatter_(1, indices_to_remove, float("-inf"))

                # розподіл після top-p
                probs = F.softmax(filtered_logits, dim=-1)

                # семпл
                next_token = torch.multinomial(probs, num_samples=1, generator=gen)

                if debug:
                    # для дебагу покажемо top-5 уже після фільтрації top-p
                    self._debug_generate(step + 1, probs, next_token.item(), top_n=5)

                # апендимо токен
                generated = torch.cat([generated, next_token], dim=-1)

                # оновлюємо attention_mask
                if attention_mask is not None:
                    pad = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
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

    def _debug_generate(self, step: int, probs: torch.Tensor, next_token_id: int, top_n: int = 5) -> None:
        """Виводить top-N ймовірностей після top-p фільтрації; позначає обраний токен."""
        topn = torch.topk(probs, k=min(top_n, probs.shape[-1]), dim=-1)
        topn_ids = topn.indices[0].tolist()
        topn_probs = topn.values[0].tolist()

        print(f"[Step {step}] (top-p filtered)")
        for i, (token_id, p_) in enumerate(zip(topn_ids, topn_probs), start=1):
            token_str = self.tokenizer.decode([token_id])
            mark = "← SELECTED" if token_id == next_token_id else ""
            print(f"  {i}. {token_str!r:20} (p={p_:.4f}) {mark}")
        print()
