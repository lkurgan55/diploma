from decoding.base_decoding import BaseStrategy
import torch

class BeamStrategy(BaseStrategy):
    """Beam decoding strategy for text generation."""

    def generate(self, prompt: str, max_new_tokens: int = 120, num_beams: int = 5, length_penalty: float = 1.05) -> str:
        """Generates text using build-in beam decoding."""
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

    def custom_generate(
        self,
        prompt: str,
        *,
        num_beams: int = 3,
        max_new_tokens: int = 64,
        length_penalty: float = 1.0,
        expand_k: int = 20,
        print_top: int = 5,
        temperature: float = 1.0,
        debug: bool = True,
    ) -> str:
        """Generates text using beam decoding with step-by-step debug output."""
        self.model.eval()

        dev = self.model.get_input_embeddings().weight.device
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_id, list):
            eos_id = eos_id[0] if eos_id else None

        out = self.model(input_ids=input_ids, use_cache=True)
        past_prompt = out.past_key_values
        logits = out.logits[:, -1, :] / max(temperature, 1e-8)

        beams: list[tuple[torch.Tensor, float, tuple]] = [(input_ids, 0.0, past_prompt)]
        finished: list[tuple[torch.Tensor, float, float]] = []

        if debug:
            print(f"\n[beam-debug] beams={num_beams} expand_k={expand_k} top_print={print_top} "
                  f"max_new_tokens={max_new_tokens} length_penalty={length_penalty} temperature={temperature}\n")

        for step in range(1, max_new_tokens + 1):
            candidates = []

            if not beams:
                break

            if debug:
                print(f"Step {step}")

            for b_idx, (seq_ids, cum_lp, past) in enumerate(beams):
                if step == 1 and b_idx == 0:
                    this_logits = logits
                else:
                    last_tok = seq_ids[:, -1:]
                    out_b = self.model(input_ids=last_tok, past_key_values=past, use_cache=True)
                    this_logits = out_b.logits[:, -1, :] / max(temperature, 1e-8)

                if debug:
                    self._debug_generate(step=step, logits=this_logits, next_token=None, top_k=print_top, beam_idx=b_idx, seq_len=seq_ids.shape[1], cum_logp=cum_lp)

                logprobs = torch.log_softmax(this_logits.squeeze(0), dim=-1)
                kexp = min(expand_k, logprobs.size(-1))
                topk_vals, topk_ids = torch.topk(logprobs, k=kexp)
                L_next = seq_ids.shape[1] + 1
                denom = (L_next ** length_penalty) if length_penalty != 0 else 1.0

                for tid, lp in zip(topk_ids.tolist(), topk_vals.tolist()):
                    new_cum = cum_lp + lp
                    norm = new_cum / denom
                    candidates.append((b_idx, tid, new_cum, norm))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[3], reverse=True)

            next_beams: list[tuple[torch.Tensor, float, tuple]] = []
            if debug:
                print("  -> chosen for next step:")

            for (parent_idx, tok_id, new_cum, norm) in candidates:
                if len(next_beams) >= num_beams:
                    break

                parent_ids, parent_cum, parent_past = beams[parent_idx]

                if eos_id is not None and tok_id == eos_id:
                    finished.append((parent_ids, new_cum, norm))
                    continue

                tok_t = torch.tensor([[tok_id]], device=parent_ids.device, dtype=parent_ids.dtype)
                new_ids = torch.cat([parent_ids, tok_t], dim=-1)

                out_next = self.model(input_ids=tok_t, past_key_values=parent_past, use_cache=True)
                new_past = out_next.past_key_values

                next_beams.append((new_ids, new_cum, new_past))

                if debug:
                    tail = self.tokenizer.decode(new_ids[0, -10:], skip_special_tokens=True)
                    print(f"     beam {len(next_beams)-1}: len={new_ids.shape[1]} cum={new_cum:.3f} norm={norm:.3f} tail='{tail}'")

            beams = next_beams
            if debug:
                print()

            if not beams:
                break

        scored = finished + [(ids, cum, (cum / (ids.shape[1] ** length_penalty)) if length_penalty != 0 else cum)
                             for (ids, cum, _) in beams]
        if not scored:
            if debug:
                print("[beam-debug] no candidates")
            return ""

        best_ids, best_cum, best_norm = max(scored, key=lambda t: t[2])
        gen_ids = best_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        if debug:
            print("[best]")
            print(f"norm={best_norm:.3f} cum={best_cum:.3f} len={best_ids.shape[1]}")
            print(text, "\n")

        return text

    def _debug_generate(
        self,
        step: int,
        logits,
        next_token: int | None = None,
        top_k: int = 5,
        *,
        beam_idx: int | None = None,
        seq_len: int | None = None,
        cum_logp: float | None = None,
    ):
        """Prints debug info for beam search at each step."""
        if logits.ndim == 2:
            logits = logits.squeeze(0)

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        k = min(top_k, logits.size(-1))
        topk_vals, topk_ids = torch.topk(logprobs, k=k)

        header = "  Beam"
        if beam_idx is not None:
            header += f" {beam_idx}"
        if seq_len is not None:
            header += f" | len={seq_len}"
        if cum_logp is not None:
            header += f" | cum={cum_logp:.3f}"
        print(header)

        for j in range(k):
            tid = topk_ids[j].item()
            token = self.tokenizer.decode([tid], skip_special_tokens=False)
            lp = topk_vals[j].item()
            p = probs[tid].item()
            mark = " ← SELECTED" if (next_token is not None and tid == next_token) else ""
            print(f"    {j+1:2d}. {repr(token):18} lp={lp:.3f} p={p:.3f}{mark}")
