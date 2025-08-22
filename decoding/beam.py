from decoding.base_decoding import BaseStrategy


class BeamStrategy(BaseStrategy):
    """
    Beam-search стратегія для генерації тексту/SQL.
    Використовує вбудований .generate з параметрами beam-пошуку.
    """

    def generate(self, prompt: str, max_new_tokens: int = 120, num_beams: int = 3, length_penalty: float = 1.05) -> str:
        """Generates text using beam decoding."""
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out = self.model.generate(
            **enc,
            do_sample=False,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def custom_generate(self, prompt: str, num_beams: int = 5, max_steps: int = 120, debug: bool = True) -> str:
        """Generates text using beam decoding with step-by-step debug output."""
        candidates: list[tuple[str, float]] = self.generate(
            prompt,
            max_new_tokens=max_steps,
            num_beams=num_beams,
        )

        if debug:
            print("\n🔎 Beam candidates:")
            for i, (txt, score) in enumerate(candidates, 1):
                short = txt.replace("\n", " ")[:220]
                print(f"{i:2d}. score={score:.4f} :: {short}")
            print()

        return candidates[0][0] if candidates else ""


    def _debug_generate(self, step: int, logits, next_token: int, top_k: int = 5):
        """Debug output for beam generation."""
        pass
