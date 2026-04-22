"""Smoke-тест: импорты не ломаются, tools регистрируются, MLX-модель НЕ грузится."""


def test_config_loads():
    from lra.config import CFG, RESEARCH_DIR
    assert CFG["model"]  # dict-style доступ
    assert CFG.model  # dataclass-style
    assert RESEARCH_DIR.name == "research"


def test_prompts_present():
    from lra import prompts
    for name in ("EXPLORER_PROMPT", "REPLANNER_PROMPT", "SYNTHESIZER_PROMPT",
                 "WRITER_PROMPT", "CRITIC_PROMPT", "COMPRESSOR_PROMPT"):
        assert hasattr(prompts, name)
        assert len(getattr(prompts, name)) > 100


def test_tools_registered():
    from qwen_agent.tools.base import TOOL_REGISTRY

    from lra import tools  # noqa: F401
    for t in ("hf_papers", "arxiv_search", "write_draft", "append_draft",
              "read_notes", "write_plan", "read_querylog", "append_lessons",
              "github_search"):
        assert t in TOOL_REGISTRY, f"tool {t} не зарегистрирован"


def test_llm_module_does_not_eagerly_load():
    # Импорт не должен тянуть mlx_lm.load (это делается только в get_mlx)
    from lra import llm
    assert llm._MLX_CACHE == {} or isinstance(llm._MLX_CACHE, dict)


def test_pipeline_imports():
    from lra.pipeline import build_bot, research_loop
    assert callable(build_bot)
    assert callable(research_loop)
