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

        self.validator = SQLValidator(db_path=db_path)
        self.validator._load_schema()

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

def efg_generate(
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
            EFGLogitsProcessor(self.tokenizer, self.validator, eos_token_id=self.tokenizer.eos_token_id)
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
    """Processor that removes invalid SQL candidates during beam search."""

    def __init__(self, tokenizer, validator: SQLValidator):
        self.tok = tokenizer
        self.validator = validator

    def __call__(self, input_ids, scores):

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            full_text = self.tok.decode(ids, skip_special_tokens=True)
            cand_sql = full_text.split("assistant", 1)[-1].strip()
            cand_sql = cand_sql.strip().strip("`")

            # table does not exist — remove candidate
            if not self.validator.tables_exist(cand_sql):
                scores[i, :] = float('-inf')

            # column does not exist — remove candidate
            if not self.validator.columns_exist(cand_sql):
                scores[i, :] = float('-inf')

            # check syntax — remove candidate
            if not self.validator.syntax_ok(cand_sql):
                scores[i, :] = float('-inf')

        return scores


class EGLALogitsProcessor(LogitsProcessor):
    """Processor that removes invalid SQL candidates during beam search (look-ahead)."""
    def __init__(self, tokenizer, validator: SQLValidator):
        self.tok = tokenizer
        self.validator = validator

    def __call__(self, input_ids, scores):

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            full_text = self.tok.decode(ids, skip_special_tokens=True)
            cand_sql = full_text.split("assistant", 1)[-1].strip()
            cand_sql = cand_sql.strip().strip("`")

            k = min(self.topk, scores.size(1))
            # take top-k candidates
            _, topk_idx = torch.topk(scores[i], k=k)

            for tid in topk_idx.tolist():
                token_str = self.tok.decode([tid], skip_special_tokens=True)
                probe = (cand_sql + token_str).strip()

                # table does not exist — remove candidate
                if not self.validator.tables_exist(probe):
                    scores[i, tid] = float('-inf')

                # column does not exist — remove candidate
                if not self.validator.columns_exist(probe):
                    scores[i, tid] = float('-inf')

                # check syntax — remove candidate
                if not self.validator.syntax_ok(probe):
                    scores[i, tid] = float('-inf')

        return scores

class EFGLogitsProcessor(LogitsProcessor):
    """Processor that removes invalid SQL candidates during beam search."""

    def __init__(self, tokenizer, validator: SQLValidator):
        self.tok = tokenizer
        self.validator = validator

    def __call__(self, input_ids, scores):

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            full_text = self.tok.decode(ids, skip_special_tokens=True)
            cand_sql = full_text.split("assistant", 1)[-1].strip()
            cand_sql = cand_sql.strip().strip("`")

            cand_sql = self.validator.sql_fix(cand_sql)

            # table does not exist — remove candidate
            if not self.validator.tables_exist(cand_sql):
                scores[i, :] = float('-inf')

            # column does not exist — remove candidate
            if not self.validator.columns_exist(cand_sql):
                scores[i, :] = float('-inf')

            # check syntax — remove candidate
            if not self.validator.syntax_ok(cand_sql):
                scores[i, :] = float('-inf')

        return scores
