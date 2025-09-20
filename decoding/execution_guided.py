# hf_parity_beam_strategy.py
from typing import List, Tuple, Optional
import inspect
import torch
from transformers.generation import (
    BeamSearchScorer,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
)
from decoding.base_decoding import BaseStrategy
from collections import UserDict
from decoding.sql_validator import SQLValidator

class EGBeamStrategy(BaseStrategy):
    """
    Відкритий beam з паритетом до HF .generate(num_beams=...),
    сумісний із Qwen2 (Cache), без prepare_inputs_for_generation на кроках.
    """

    @staticmethod
    def _postprocess(text: str, ensure_semicolon: bool = True) -> str:
        t = (text or "").strip()
        if ensure_semicolon and t and not t.endswith(";"):
            t += ";"
        return t

    # ---------- універсальний reorder кешу ----------
    def _reorder_cache_safe(self, past, beam_idx: torch.LongTensor):
        if past is None:
            return None
        if hasattr(self.model, "_reorder_cache"):
            return self.model._reorder_cache(past, beam_idx)

        def _rec(x):
            # деякі Cache-типи мають власний index_select
            if hasattr(x, "index_select") and not isinstance(x, torch.Tensor):
                try:
                    return x.index_select(beam_idx)
                except TypeError:
                    try:
                        return x.index_select(0, beam_idx)
                    except Exception:
                        pass
            if isinstance(x, torch.Tensor):
                return x.index_select(0, beam_idx)
            if isinstance(x, dict):
                return {k: _rec(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(_rec(v) for v in x)
            return x

        return _rec(past)

    # ---------- перший форвард: повна послідовність, без past ----------
    def _first_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        return out.logits[:, -1, :], out.past_key_values  # Cache

    # ---------- кроковий форвард: лише останній токен + Cache ----------
    def _step_forward(self, last_token: torch.Tensor, past, attention_mask: Optional[torch.Tensor]):
        if attention_mask is None:
            attention_mask = torch.ones((last_token.size(0), 1), dtype=torch.long, device=last_token.device)
        with torch.no_grad():
            out = self.model(
                input_ids=last_token,      # [B, 1]
                past_key_values=past,      # Cache-об’єкт, не tuple!
                attention_mask=attention_mask,
                use_cache=True,
            )
        return out.logits[:, -1, :], out.past_key_values

    # ---------- сумісний фіналізатор для різних версій HF ----------
    def _finalize_sequences(
        self,
        scorer: BeamSearchScorer,
        seq_input_ids: torch.Tensor,
        beam_scores: torch.Tensor,
        *,
        max_length: int,
        pad_token_id: int,
        eos_token_id: Optional[int],
    ):
        """
        Підтримує 3 API:
          - старий:  finalize(input_ids, beam_scores, next_tokens, next_indices, max_length, pad_token_id, eos_token_id)
          - проміжний: finalize(input_ids, beam_scores, pad_token_id=..., eos_token_id=..., max_length=...)
          - новий:  finalize(input_ids, final_beam_scores=..., pad_token_id=..., eos_token_id=..., max_length=..., beam_indices=?)
        """
        sig = inspect.signature(scorer.finalize)
        names = list(sig.parameters.keys())

        try:
            if "final_beam_scores" in names:
                # новий API: ПЕРЕДАЄМО БАЛИ ЛИШЕ КЛЮЧЕМ final_beam_scores!
                kwargs = dict(
                    final_beam_scores=beam_scores,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    max_length=max_length,
                )
                if "beam_indices" in names:
                    kwargs["beam_indices"] = None
                out = scorer.finalize(seq_input_ids, **kwargs)
            elif ("next_tokens" in names) or ("next_indices" in names):
                # старий API
                out = scorer.finalize(
                    seq_input_ids,
                    beam_scores,
                    None,  # next_tokens
                    None,  # next_indices
                    max_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            else:
                # проміжний API (beam_scores як другий позиційний ок)
                out = scorer.finalize(
                    seq_input_ids,
                    beam_scores,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    max_length=max_length,
                )
        except TypeError:
            # страховки
            try:
                out = scorer.finalize(
                    seq_input_ids,
                    beam_scores,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    max_length=max_length,
                )
            except TypeError:
                out = scorer.finalize(
                    seq_input_ids,
                    beam_scores,
                    None,
                    None,
                    max_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

        if isinstance(out, (dict, UserDict)):
            return out["sequences"]

        # 2) об'єкт із атрибутом sequences
        if hasattr(out, "sequences"):
            return out.sequences

        # 3) інколи UserDict лежить у .data
        if hasattr(out, "data") and isinstance(out.data, (dict, UserDict)) and "sequences" in out.data:
            return out.data["sequences"]

        # 4) крайній випадок — якщо це список/кортеж тензорів/списків, спробуємо зібрати тензор
        if isinstance(out, (list, tuple)):
            try:
                return torch.stack([torch.as_tensor(x) for x in out], dim=0)
            except Exception:
                pass

        # якщо сюди дійшли — віддай як є (щоб бачити тип у дебазі)
        return out


    # ---------- простий виклик через .generate ----------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        ensure_semicolon: bool = True,
    ) -> str:
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # нормалізуємо pad_id і eos_id
        eos_id = self.tokenizer.eos_token_id or getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            getattr(getattr(self.model, "generation_config", None), "pad_token_id", None)
        )
        if pad_id is None:
            pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        if pad_id is None:
            raise ValueError("pad_token_id is not defined; set tokenizer.pad_token_id or model.config.pad_token_id.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = pad_id

        out = self.model.generate(
            **enc,
            do_sample=False,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._postprocess(text, ensure_semicolon=ensure_semicolon)

    # ---------- top-1 через «відкритий» beam ----------
    def custom_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        ensure_semicolon: bool = True,
        debug: bool = False,
        print_top: int = 5,
        db_path: str | None = None,
    ) -> str:
        if db_path:
            self.validator = SQLValidator(db_path=db_path)
            self.validator._maybe_load_schema()

        cands, _ = self.preview_beam(
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            ensure_semicolon=ensure_semicolon,
            return_scores=False,
            debug=debug,
            print_top=print_top,
        )
        return cands[0] if cands else ""

    # ---------- відкритий beam з паритетом до HF ----------
    def preview_beam(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        ensure_semicolon: bool = True,
        return_scores: bool = True,
        debug: bool = True,
        print_top: int = 5,
    ) -> Tuple[List[str], Optional[List[float]]]:

        self.model.eval()

        # 1) encode
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids0 = enc["input_ids"].to(self.model.device)
        attention_mask0 = enc.get("attention_mask")
        if attention_mask0 is None:
            attention_mask0 = torch.ones_like(input_ids0, dtype=torch.long, device=self.model.device)
        else:
            attention_mask0 = attention_mask0.to(self.model.device)

        # нормалізуємо eos і pad
        eos_token_id = self.tokenizer.eos_token_id or getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            getattr(getattr(self.model, "generation_config", None), "pad_token_id", None)
        )
        if pad_id is None:
            pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_token_id
        if pad_id is None:
            raise ValueError("pad_token_id is not defined; set tokenizer.pad_token_id or model.config.pad_token_id.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = pad_id

        batch_size = input_ids0.size(0)
        assert batch_size == 1, "Для простоти: 1 запит за раз."

        # 2) processors
        gc = getattr(self.model, "generation_config", None)
        rep_pen = getattr(gc, "repetition_penalty", 1.0) if gc else 1.0
        ngram = getattr(gc, "no_repeat_ngram_size", 0) if gc else 0
        min_len = getattr(gc, "min_length", 0) if gc else 0

        processors = LogitsProcessorList()
        if rep_pen and rep_pen != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(rep_pen))
        if ngram and ngram > 0:
            processors.append(NoRepeatNGramLogitsProcessor(ngram))
        if min_len and min_len > 0 and eos_token_id is not None:
            processors.append(MinLengthLogitsProcessor(min_len, eos_token_id))

        # 3) BeamSearchScorer
        max_length = input_ids0.shape[1] + max_new_tokens
        scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=self.model.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_beams,
        )

        # 4) дублюємо prompt для бімів
        seq_input_ids = input_ids0.expand(num_beams, -1)         # для scorer/finalize
        model_input_ids = seq_input_ids                          # спочатку подаємо повні
        attention_mask = attention_mask0.expand(num_beams, -1)   # маска точно є

        # початкові бім-скори
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.model.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        cur_len = seq_input_ids.shape[-1]
        past = None  # Cache
        enc_len = input_ids0.shape[1]

        if debug:
            print(f"\n[hf-parity-beam] beams={num_beams} max_new={max_new_tokens} lp={length_penalty}\n")

        # 5) декодування
        while cur_len < max_length:
            # перший крок: повні ids; далі — тільки останній токен
            if past is None:
                logits, past = self._first_forward(model_input_ids, attention_mask)
            else:
                last_token = model_input_ids[:, -1:]   # [num_beams, 1]
                logits, past = self._step_forward(last_token, past, attention_mask)

            # лог-простір + процесори
            next_token_scores = torch.log_softmax(logits, dim=-1)
            next_token_scores = processors(seq_input_ids, next_token_scores)

            # додаємо beam_scores
            next_token_scores = next_token_scores + beam_scores[:, None]

            # вибір top 2*num_beams
            vocab_size = next_token_scores.size(-1)
            flat_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            topk = torch.topk(flat_scores, 2 * num_beams, dim=1)
            next_token_scores = topk.values
            next_tokens = topk.indices % vocab_size
            next_indices = topk.indices // vocab_size

            # EG: перевірка на валідність SQL
            if self.validator:
                def _decode(ids_row: torch.Tensor) -> str:
                    full_text = self.tokenizer.decode(ids_row.tolist(), skip_special_tokens=True)
                    sql = full_text.split("assistant", 1)[-1].strip()
                    return sql.strip().strip("`")

                for j in range(next_tokens.size(1)):  # 2*num_beams кандидатів
                    pidx = int(next_indices[0, j].item())
                    cand_ids = torch.cat([seq_input_ids[pidx], next_tokens[0, j].unsqueeze(0)], dim=0)
                    cand_sql = _decode(cand_ids)

                    try:
                        # перевірка синтаксису — знімаємо кандидата
                        # if not self.validator.syntax_ok(cand_sql):
                        #     next_token_scores[0, j] = -1e9
                        # # якась таблиця не існує — знімаємо кандидата
                        # if not self.validator.tables_exist(cand_sql):
                        #     next_token_scores[0, j] = -1e9
                        # # якась колонка не існує — знімаємо кандидата
                        # if not self.validator.columns_exist(cand_sql):
                        #     next_token_scores[0, j] = -1e9
                        pass
                    except Exception:
                        # у разі збою валідації — не ріжемо кандидата
                        pass

            # --- EG END ---

            # процес відбору HF
            next_beam = scorer.process(
                seq_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_id,           # <— використовуємо pad_id
                eos_token_id=eos_token_id,     # int
                beam_indices=None,
            )
            beam_scores = next_beam["next_beam_scores"]
            beam_next_tokens = next_beam["next_beam_tokens"]
            beam_next_indices = next_beam["next_beam_indices"]

            # оновлюємо послідовності (для фіналізації)
            seq_input_ids = torch.cat([seq_input_ids[beam_next_indices, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # для наступного кроку в модель подаємо тільки останній токен
            model_input_ids = beam_next_tokens.unsqueeze(-1)

            # reorder кеш і маску
            beam_next_indices = beam_next_indices.to(self.model.device, dtype=torch.long)
            past = self._reorder_cache_safe(past, beam_next_indices)
            attention_mask = torch.cat(
                [
                    attention_mask[beam_next_indices, :],
                    torch.ones((num_beams, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=-1,
            )

            cur_len += 1

            if debug:
                lp = torch.log_softmax(logits[0], dim=-1)
                probs = torch.softmax(logits[0], dim=-1)
                k = min(print_top, logits.shape[-1])
                vals, ids = torch.topk(lp, k=k)
                print(f"Step {cur_len - enc_len}")
                for j in range(k):
                    tok = self.tokenizer.decode([ids[j].item()], skip_special_tokens=False)
                    print(f"  {j+1}. {tok!r:18} lp={vals[j].item():.3f} p={probs[ids[j]].item():.3f}")
                print()

            if scorer.is_done or cur_len >= max_length:
                break

        # 6) фіналізація (універсальна)
        seqs = self._finalize_sequences(
            scorer,
            seq_input_ids,
            beam_scores,
            max_length=max_length,
            pad_token_id=pad_id,           # <— використовуємо pad_id
            eos_token_id=eos_token_id,     # int
        )

        # 7) розшифровуємо
        texts = []
        for i in range(min(num_beams, seqs.size(0))):
            gen_ids = seqs[i][enc_len:]
            txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            texts.append(self._postprocess(txt, ensure_semicolon=ensure_semicolon))

        return texts, (None if not return_scores else [0.0] * len(texts))
