"""MLX-бэкенд для qwen-agent. Модель кешируется глобально."""
from __future__ import annotations
from typing import Dict, Iterator, List, Optional

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message

from .config import CFG

# Глобальный кеш весов — чтобы несколько Assistant не грузили модель дважды
_MLX_CACHE: Dict[str, tuple] = {}


def get_mlx(model_name: str):
    if model_name not in _MLX_CACHE:
        from mlx_lm import load
        _MLX_CACHE[model_name] = load(model_name)
    return _MLX_CACHE[model_name]


@register_llm("mlx")
class MlxLLM(BaseFnCallModel):
    """Локальный MLX-бэкенд для Qwen-Agent."""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.mlx_model, self.tokenizer = get_mlx(cfg["model"])

    def _build_prompt(self, messages: List[Message]) -> str:
        plain = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in messages]
        return self.tokenizer.apply_chat_template(
            plain, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )

    def _mlx_generate(self, prompt: str, cfg: dict):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        sampler = make_sampler(
            temp=cfg.get("temperature", CFG["temperature"]),
            top_p=cfg.get("top_p", CFG["top_p"]),
            top_k=cfg.get("top_k", CFG["top_k"]),
        )
        proc = make_logits_processors(repetition_penalty=CFG["repetition_penalty"])
        return stream_generate(
            self.mlx_model, self.tokenizer, prompt=prompt,
            max_tokens=cfg.get("max_tokens", CFG["max_tokens"]),
            sampler=sampler, logits_processors=proc,
        )

    def _chat_stream(self, messages, delta_stream, generate_cfg) -> Iterator[List[Message]]:
        prompt = self._build_prompt(messages)
        acc = ""
        for resp in self._mlx_generate(prompt, generate_cfg):
            if delta_stream:
                yield [Message(ASSISTANT, resp.text)]
            else:
                acc += resp.text
                yield [Message(ASSISTANT, acc)]

    def _chat_no_stream(self, messages, generate_cfg) -> List[Message]:
        prompt = self._build_prompt(messages)
        text = "".join(r.text for r in self._mlx_generate(prompt, generate_cfg))
        return [Message(ASSISTANT, text)]
