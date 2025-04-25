import openai
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
        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        model_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "models/bge-large-zh"
        logger.info(f"模型路径: {model_path}")
        # 初始化 HuggingFace Embedding
        embedding_model = HuggingFaceEmbeddings(
            model_name=str(model_path),
            # model_name="BAAI/bge-base-zh",
            model_kwargs={"device": "cpu"}
        )
        logger.info("Embedding 模型加载完成")

        # 加载本地 FAISS 向量库
        faiss_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "data/faiss_index_DeepSeek"
        if not faiss_path.exists():
            raise FileNotFoundError(f"未找到向量库: {faiss_path}")
        
        self.vectorstore = FAISS.load_local(
            str(faiss_path),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # 构建 RAG 检索链
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
        logger.info(f"LLMProvider 初始化完成")

    def response(self, session_id, dialogue):
        query = dialogue[-1]["content"] if dialogue else ""
        if not query.strip():
            return "请输入问题。"

        if self._is_knowledge_query(query):
            logger.info(f"[回答来源] 使用 RAG 模型回答: {query}")
            return self.rag_response(query)
        else:
            logger.info(f"[回答来源] 使用 OpenAI 模型回答: {query}")
            return self.openai_response(dialogue)

    def _is_knowledge_query(self, query: str) -> bool:
        return any(kw in query for kw in KNOWLEDGE_KEYWORDS)

    def rag_response(self, query: str) -> str:
        try:
            result = self.qa_chain.invoke({"query": query})
            return result["result"]
        except Exception as e:
            logger.bind(tag=TAG).error(f"RAG模型出错: {e}")
            return "RAG模型处理失败"

    def openai_response(self, session_id, dialogue):
        try:
            responses = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=False,
                max_tokens=self.max_tokens,
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
            stream = self.client.chat.completions.create(
                model=self.model_name, messages=dialogue, stream=True, tools=functions
            )

            for chunk in stream:
                # 检查是否存在有效的choice且content不为空
                if getattr(chunk, "choices", None):
                    yield chunk.choices[0].delta.content, chunk.choices[0].delta.tool_calls
                # 存在 CompletionUsage 消息时，生成 Token 消耗 log
                elif isinstance(getattr(chunk, 'usage', None), CompletionUsage):
                    usage_info = getattr(chunk, 'usage', None)
                    logger.bind(tag=TAG).info(
                        f"Token 消耗：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，" 
                        f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                        f"共计 {getattr(usage_info, 'total_tokens', '未知')}"
                    )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in function call streaming: {e}")
            yield f"【OpenAI服务响应异常: {e}】", None
