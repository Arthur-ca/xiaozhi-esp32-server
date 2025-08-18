import httpx
import openai
import os, yaml
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase
from core.providers.llm.rag.pipeline import RAGPipeline

TAG = __name__
logger = setup_logging()
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")

class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        if "base_url" in config:
            self.base_url = config.get("base_url")
        else:
            self.base_url = config.get("url")
        # 增加timeout的配置项，单位为秒
        timeout = config.get("timeout", 300)
        self.timeout = int(timeout) if timeout else 300

        param_defaults = {
            "max_tokens": (500, int),
            "temperature": (0.7, lambda x: round(float(x), 1)),
            "top_p": (1.0, lambda x: round(float(x), 1)),
            "frequency_penalty": (0, lambda x: round(float(x), 1)),
        }

        for param, (default, converter) in param_defaults.items():
            value = config.get(param)
            try:
                setattr(
                    self,
                    param,
                    converter(value) if value not in (None, "") else default,
                )
            except (ValueError, TypeError):
                setattr(self, param, default)

        logger.debug(
            f"意图识别参数初始化: {self.temperature}, {self.max_tokens}, {self.top_p}, {self.frequency_penalty}"
        )

        model_key_msg = check_model_key("LLM", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=httpx.Timeout(self.timeout))

         # === 新增：加载 RAG 配置并初始化 Pipeline（仅 functions 路线用） ===
        rag_cfg = {}
        cfg_path = os.path.join(os.getcwd(), "config", "rag_config.yaml")
        if os.path.exists(cfg_path):
            try: rag_cfg = (yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {})
            except Exception: rag_cfg = {}
        self._rag = RAGPipeline(rag_cfg.get("rag", rag_cfg))

        # 一个内部流式封装，供 RAG 使用
    def _stream_text(self, messages, **overrides):
        stream = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True,
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            temperature=overrides.get("temperature", self.temperature),
            top_p=overrides.get("top_p", self.top_p),
            frequency_penalty=overrides.get("frequency_penalty", self.frequency_penalty),
        )
        for chunk in stream:
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                txt = getattr(delta, "content", "") if delta else ""
                if txt: yield txt

    def response(self, session_id, dialogue, **kwargs):
        try:
            responses = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get(
                    "frequency_penalty", self.frequency_penalty
                ),
            )

            is_active = True
            for chunk in responses:
                try:
                    # 检查是否存在有效的choice且content不为空
                    delta = (
                        chunk.choices[0].delta
                        if getattr(chunk, "choices", None)
                        else None
                    )
                    content = delta.content if hasattr(delta, "content") else ""
                except IndexError:
                    content = ""
                if content:
                    # 处理标签跨多个chunk的情况
                    if "<think>" in content:
                        is_active = False
                        content = content.split("<think>")[0]
                    if "</think>" in content:
                        is_active = True
                        content = content.split("</think>")[-1]
                    if is_active:
                        yield content

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in response generation: {e}")

    def response_with_functions(self, session_id, dialogue, functions=None):
        try:
            if getattr(self, "_rag", None) and self._rag.enabled:
                # 可选：若需要 MQR，用一个轻量 LLM 做 query 扩展（复用当前 base_url/api_key）
                mqr_llm = {"model": self.model_name, "base_url": self.base_url, "api_key": self.api_key}
                # RAGPipeline 会判断是否足够“有依据”；否则直接把原 dialogue 交回（相当于 FUNCTION 路线前置）
                for text in self._rag.stream(dialogue, lambda msgs: self._stream_text(msgs), mqr_llm=mqr_llm):
                    yield text, None
                return

            stream = self.client.chat.completions.create(
                model=self.model_name, messages=dialogue, stream=True, tools=functions
            )

            for chunk in stream:
                # 检查是否存在有效的choice且content不为空
                if getattr(chunk, "choices", None):
                    yield chunk.choices[0].delta.content, chunk.choices[
                        0
                    ].delta.tool_calls
                # 存在 CompletionUsage 消息时，生成 Token 消耗 log
                elif isinstance(getattr(chunk, "usage", None), CompletionUsage):
                    usage_info = getattr(chunk, "usage", None)
                    logger.bind(tag=TAG).info(
                        f"Token 消耗：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，"
                        f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                        f"共计 {getattr(usage_info, 'total_tokens', '未知')}"
                    )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in function call streaming: {e}")
            yield f"【OpenAI服务响应异常: {e}】", None
