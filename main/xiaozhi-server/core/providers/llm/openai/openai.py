import openai
import re
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from pathlib import Path

TAG = __name__
logger = setup_logging()

# 关键词触发列表
KNOWLEDGE_KEYWORDS = [
    "孕妇", "怀孕", "妊娠", "胎儿", "产前", "孕期", "产检", "胎心", "唐筛", "孕期营养", "孕妇饮食", "孕期运动", "孕期症状"
]

class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        if "base_url" in config:
            self.base_url = config.get("base_url")
        else:
            self.base_url = config.get("url")
        max_tokens = config.get("max_tokens")
        if max_tokens is None or max_tokens == "":
            max_tokens = 500
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            max_tokens = 500
        self.max_tokens = max_tokens
        check_model_key("LLM", self.api_key)
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 初始化 RAG 模型
        model_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "models/bge-large-zh"
        embedding_model = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"device": "cpu"}
        )
        logger.info("[RAG] Embedding 模型加载完成")

        faiss_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "data/faiss_index_DeepSeek"
        if not faiss_path.exists():
            raise FileNotFoundError(f"未找到向量库: {faiss_path}")

        self.vectorstore = FAISS.load_local(
            str(faiss_path),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                temperature=0.2,
            ),
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            return_source_documents=True,
        )
        logger.info("[RAG] 检索链初始化完成")

    def _is_knowledge_query(self, query: str) -> bool:
        matched_keywords = [kw for kw in KNOWLEDGE_KEYWORDS if kw in query]
        logger.info(f"[RAG-FILTER] 匹配关键词: {matched_keywords}")
        return bool(matched_keywords)
    
    def clean_rag_text(self, text: str) -> str:
        """清理RAG输出中的Markdown符号,让TTS更自然"""
        text = re.sub(r'#', '', text)  # 去掉 #
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去掉 **加粗**
        text = re.sub(r'^\s*-\s*', '', text, flags=re.MULTILINE)  # 去掉每行开头的 -
        text = re.sub(r'\*', '', text)  # 去掉孤立的 *
        text = re.sub(r'\n{2,}', '\n', text)  # 多个连续换行变一个
        return text.strip()

    def rag_response_stream(self, query: str):
        logger.info(f"[RAG-STREAM] 开始流式处理查询: {query}")
        try:
            result = self.qa_chain.invoke({"query": query})
            full_answer = result.get("result", "")
            logger.info("[RAG-STREAM] 模型调用成功")

            cleaned_text = self.clean_rag_text(full_answer)

            buffer = ""
            for sentence in re.split(r'(。|！|\!|\\?|\\？)', cleaned_text):
                if sentence.strip():
                    buffer += sentence
                    if len(buffer) >= 100:
                        yield buffer.strip(), None
                        buffer = ""

            # 输出剩余的
            if buffer.strip():
                yield buffer.strip(), None

        except Exception as e:
            logger.error(f"[RAG-STREAM] 流式处理失败: {e}")
            yield "【RAG模型处理失败】", None

    def response(self, session_id, dialogue):
        logger.info("[OPENAI] 调用 response")
        try:
            responses = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True,
                max_tokens=self.max_tokens,
            )
            is_active = True
            for chunk in responses:
                try:
                    delta = (
                        chunk.choices[0].delta
                        if getattr(chunk, "choices", None)
                        else None
                    )
                    content = delta.content if hasattr(delta, "content") else ""
                except IndexError:
                    content = ""
                if content:
                    if "<think>" in content:
                        is_active = False
                        content = content.split("<think>")[0]
                    if "</think>" in content:
                        is_active = True
                        content = content.split("</think>")[-1]
                    if is_active:
                        yield content
        except Exception as e:
            logger.error(f"[OPENAI] response 模式失败: {e}")

    def response_with_functions(self, session_id, dialogue, functions=None):
        logger.info("[OPENAI] 调用 response_with_functions")
        try:
            query = dialogue[-1]["content"] if dialogue else ""
            logger.info(f"[OPENAI] 收到请求: {query}")

            if self._is_knowledge_query(query):
                logger.info("[OPENAI] 命中关键词，使用 RAG 流式模型")
                for chunk, _ in self.rag_response_stream(query):
                    yield chunk, None
                return

            logger.info("[OPENAI] 未命中关键词，使用 function 模式")
            stream = self.client.chat.completions.create(
                model=self.model_name, messages=dialogue, stream=True, tools=functions
            )

            for chunk in stream:
                if getattr(chunk, "choices", None):
                    yield chunk.choices[0].delta.content, chunk.choices[0].delta.tool_calls
                elif isinstance(getattr(chunk, 'usage', None), CompletionUsage):
                    usage_info = getattr(chunk, 'usage', None)
                    logger.info(
                        f"[OPENAI] Token 使用情况：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，"
                        f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                        f"总计 {getattr(usage_info, 'total_tokens', '未知')}"
                    )
        except Exception as e:
            logger.error(f"[OPENAI] Function 模式调用失败: {e}")
            yield f"【OpenAI服务响应异常: {e}】", None