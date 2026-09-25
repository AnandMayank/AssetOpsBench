"""OpenAI-compatible LLM backend (no litellm dependency).

For gateways that expose the standard OpenAI Chat Completions API — such as
`TokenRouter <https://www.tokenrouter.com>`_ — we talk to them with the
``openai`` SDK directly instead of routing through litellm.  litellm only
earns its keep for providers that are *not* OpenAI-shaped (e.g. watsonx).

The prefix→endpoint mapping lives in :mod:`llm.routers` (shared with the
agent runners).  The bare model name is sent to the endpoint::

    tokenrouter/MiniMax-M3   →  POST {TOKENROUTER_BASE_URL}/chat/completions
                                with model="MiniMax-M3"
"""

from __future__ import annotations

from .base import LLMBackend, LLMResult
from .routers import is_openai_compat, resolve_model, resolve_router_creds

__all__ = ["OpenAICompatBackend", "is_openai_compat"]

#: Per-model token-budget override, a disclosed protocol deviation -- NOT
#: a global change. deepseek/deepseek-v4-pro-0813 verified live to
#: exhaust the standard 2048-token budget on hidden reasoning before
#: producing any visible output, 100% of the time on D-physical's
#: single-turn protocol (70/70 episodes, output_tokens==2048,
#: text==""). Every other model is unaffected. Checked by prefixed id
#: (this backend's model_id) since that is what every caller passes.
MODEL_TOKEN_OVERRIDES = {
    "tokenrouter/deepseek/deepseek-v4-pro-0813": 16384,
}


class OpenAICompatBackend(LLMBackend):
    """LLM backend using the native ``openai`` SDK against a compatible router.

    Args:
        model_id: prefixed model string, e.g. ``"tokenrouter/MiniMax-M3"``.
    """

    def __init__(self, model_id: str) -> None:
        if not is_openai_compat(model_id):
            raise ValueError(
                f"unsupported OpenAI-compatible model id: {model_id!r}"
            )
        self._model_id = model_id
        self._model_name = resolve_model(model_id)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0
    ) -> LLMResult:
        """Two disclosed, model-triggered protocol fallbacks, verified
        live at the GPT-6-Astra feasibility check (same fix as the raw-
        urllib _chat() helpers' identical logic -- see
        run_l3_pilot_executed._chat's docstring for the exact error text):
        some models reject 'max_tokens' (-> retry as 'max_completion_tokens')
        and/or reject temperature=0 (-> retry with temperature omitted,
        a real deviation from the rest of the panel's protocol that any
        caller must disclose, not silently absorb). Every model that
        accepts the standard request never hits either fallback."""
        from openai import BadRequestError, OpenAI

        creds = resolve_router_creds(self._model_id)  # strict: clear error if unset
        client = OpenAI(base_url=creds.base_url, api_key=creds.api_key)

        max_tokens = MODEL_TOKEN_OVERRIDES.get(self._model_id, 2048)
        tokens_kwarg, include_temperature = "max_tokens", True
        response = None
        for _attempt in range(3):
            kwargs: dict = {"model": self._model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            tokens_kwarg: max_tokens}
            if include_temperature:
                kwargs["temperature"] = temperature
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except BadRequestError as exc:
                body_text = str(exc)
                if "max_tokens" in body_text and "max_completion_tokens" in body_text and tokens_kwarg == "max_tokens":
                    tokens_kwarg = "max_completion_tokens"
                    continue
                if "temperature" in body_text and include_temperature:
                    include_temperature = False
                    continue
                raise
        if response is None:
            raise RuntimeError(f"{self._model_id}: exhausted protocol-fallback retries")
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=response.choices[0].message.content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            served_model=getattr(response, "model", None),
        )
