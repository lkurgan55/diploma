# from decoding.base_decoding import BaseStrategy
# import torch

# class GreedyStrategy(BaseStrategy):
#     """Greedy decoding strategy for text generation."""

#     def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
#         """Generates text using build-in greedy decoding."""
#         self.model.eval()
#         enc = self.tokenizer(prompt, return_tensors="pt")
#         enc = {k: v.to(self.model.device) for k, v in enc.items()}

#         out = self.model.generate(
#             **enc,
#             do_sample=False,
#             pad_token_id=self.tokenizer.eos_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             max_new_tokens=max_new_tokens,
#         )

#         gen_ids = out[0][enc["input_ids"].shape[1]:]
#         return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

#     def custom_generate(self, prompt: str, debug: bool = False, max_steps: int = 50) -> str:
#         """Generates text using greedy decoding with step-by-step debug output."""
#         self.model.eval()
#         enc = self.tokenizer(prompt, return_tensors="pt")
#         input_ids = enc["input_ids"].to(self.model.device)
#         attention_mask = enc.get("attention_mask")
#         if attention_mask is not None:
#             attention_mask = attention_mask.to(self.model.device)

#         generated = input_ids.clone()
#         eos_id = self.tokenizer.eos_token_id or getattr(self.model.config, "eos_token_id", None)
#         step = 0

#         if debug:
#             print("\n🔎 DEBUG:\n")

#         with torch.no_grad():
#             while True:
#                 outputs = self.model(input_ids=generated, attention_mask=attention_mask)
#                 logits = outputs.logits[:, -1, :]

#                 next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

#                 if debug:
#                     self._debug_generate(step + 1, logits, next_token.item(), top_k=5)

#                 generated = torch.cat([generated, next_token], dim=-1)

#                 if attention_mask is not None:
#                     pad = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
#                     attention_mask = torch.cat([attention_mask, pad], dim=-1)

#                 step += 1

#                 if eos_id is not None and next_token.item() == eos_id:
#                     if debug:
#                         print("🔚 Found EOS.\n")
#                     break
#                 if step >= max_steps:
#                     if debug:
#                         print(f"⚠️ Limit {max_steps} tokens.")
#                     break

#         gen_ids = generated[0, input_ids.shape[1]:]
#         return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

#     def _debug_generate(self, step: int, logits: torch.Tensor, next_token_id: int, top_k: int = 5) -> None:
#         """Prints the top-k token probabilities for the current step."""
#         probs = torch.softmax(logits, dim=-1)
#         topk = torch.topk(probs, k=top_k)
#         topk_ids = topk.indices[0].tolist()
#         topk_probs = topk.values[0].tolist()

#         print(f"[Step {step}]")
#         for i, (token_id, prob) in enumerate(zip(topk_ids, topk_probs)):
#             token_str = self.tokenizer.decode([token_id])
#             mark = "← SELECTED" if token_id == next_token_id else ""
#             print(f"  {i+1}. {token_str!r:20} (p={prob:.4f}) {mark}")

#         print()
