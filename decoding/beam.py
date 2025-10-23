from decoding.base_decoding import BaseStrategy
from decoding.sql_validator import SQLValidator
from transformers import LogitsProcessor, LogitsProcessorList
import torch

class BeamStrategy(BaseStrategy):
    """Beam decoding strategy for text generation with HF-parity custom loop."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        num_beams: int = 5,
        length_penalty: float = 1.05,
    ) -> str:
        """Generates text using built-in HF beam decoding."""
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


    def eg_generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        num_beams: int = 5,
        db_path: str | None = None
    ) -> str:
        if db_path:
            self.validator = SQLValidator(db_path=db_path)
            self.validator._maybe_load_schema()
            
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        processors = LogitsProcessorList([
            EGLogitsProcessor(self.tokenizer, self.validator, eos_token_id=self.tokenizer.eos_token_id)
        ])

        out = self.model.generate(
            **enc,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=processors,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_ids = out.sequences[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def egla_generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        num_beams: int = 5,
        db_path: str | None = None
    ) -> str:
        if db_path:
            self.validator = SQLValidator(db_path=db_path)
            self.validator._maybe_load_schema()
            
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        processors = LogitsProcessorList([
            EGLALogitsProcessor(self.tokenizer, self.validator, eos_token_id=self.tokenizer.eos_token_id)
        ])

        out = self.model.generate(
            **enc,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=processors,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_ids = out.sequences[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

class EGLogitsProcessor(LogitsProcessor):
    """
    Якщо beam-префікс вже невалідний (схема), відрізаємо цей бім: виставляємо -inf 
    всім логітам (beam буде вибитий з конкурсу).
    """
    def __init__(self, tokenizer, validator: SQLValidator, eos_token_id: int):
        self.tok = tokenizer
        self.validator = validator
        self.eos_id = eos_token_id
        self.dead_cache = set()  # хеші/рядки префіксів, уже позначених як мертві

    def __call__(self, input_ids, scores):
        # input_ids: [batch_beam, seq_len]
        # scores:    [batch_beam, vocab]

        # print()
        # for i in range(5):
        #     full_text = self.tok.decode(input_ids[i].tolist(), skip_special_tokens=True)
        #     cand_sql = full_text.split("assistant", 1)[-1].strip()
        #     cand_sql = cand_sql.strip().strip("`")
        #     print(cand_sql)

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            
            full_text = self.tok.decode(ids, skip_special_tokens=True)
            cand_sql = full_text.split("assistant", 1)[-1].strip()
            cand_sql = cand_sql.strip().strip("`")

            try:
                # якась таблиця не існує — знімаємо кандидата
                if not self.validator.tables_exist(cand_sql):
                    scores[i, :] = float('-inf')

                # # якась колонка не існує — знімаємо кандидата
                if not self.validator.columns_exist(cand_sql):
                    scores[i, :] = float('-inf')

                # перевірка синтаксису — знімаємо кандидата
                if not self.validator.syntax_ok(cand_sql):
                    scores[i, :] = float('-inf')
                pass
            except Exception:
                # у разі збою валідації — не ріжемо кандидата
                pass

        return scores


class EGLALogitsProcessor(LogitsProcessor):
    """
    Execution-Guided Look-Ahead:
    На кроці t для кожного біма дивимося top-k наступних токенів і м'яко штрафуємо ті,
    що ведуть до явно поганого префікса (схема/синтаксис), поки префікс ще незавершений.
    """
    def __init__(self, tokenizer, validator: SQLValidator, eos_token_id: int):
        self.tok = tokenizer
        self.validator = validator
        self.eos_id = eos_token_id
        self.dead_cache = set()  # хеші/рядки префіксів, уже позначених як мертві

        self.penalty = 4.0
        self.topk = 32

    def __call__(self, input_ids, scores):
        # input_ids: [batch_beam, seq_len]
        # scores:    [batch_beam, vocab]

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            
            full_text = self.tok.decode(ids, skip_special_tokens=True)
            cand_sql = full_text.split("assistant", 1)[-1].strip()
            cand_sql = cand_sql.strip().strip("`")

            k = min(self.topk, scores.size(1))
            # беремо індекси top-k і дивимось, що буде, якщо додати кожен токен
            _, topk_idx = torch.topk(scores[i], k=k)

            for tid in topk_idx.tolist():
                token_str = self.tok.decode([tid], skip_special_tokens=True)
                probe = (cand_sql + token_str).strip()
                
                try:
                    # якась таблиця не існує — знімаємо кандидата
                    if not self.validator.tables_exist(probe):
                        scores[i, tid] = float('-inf')

                    # # якась колонка не існує — знімаємо кандидата
                    if not self.validator.columns_exist(probe):
                        scores[i, tid] = float('-inf')

                    # перевірка синтаксису — знімаємо кандидата
                    if not self.validator.syntax_ok(probe):
                        scores[i, tid] = float('-inf')
                    pass
                except Exception:
                    # у разі збою валідації — не ріжемо кандидата
                    pass

        return scores