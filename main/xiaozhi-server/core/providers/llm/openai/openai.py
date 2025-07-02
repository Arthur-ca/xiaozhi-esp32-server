import httpx
import openai
import re
import time
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pathlib import Path

TAG = __name__
logger = setup_logging()

# 关键词触发列表
KNOWLEDGE_KEYWORDS = [
    "孕妇", "怀孕", "妊娠", "胎儿", "产前", "孕期", "产检", "胎心", "唐筛", "孕期营养", "孕妇饮食", "孕期运动", "孕期症状"
]

# 优化1: 自定义RAG提示词模板，提供更明确的指导
RAG_PROMPT_TEMPLATE = """
请基于以下参考信息回答用户的问题。
如果参考信息中没有相关内容，请直接说明您不知道，不要编造信息。
回答要简洁、准确、有帮助性，并直接针对用户问题给出答案。

参考信息:
{context}

用户问题: {question}

回答:
"""

# 创建优化的提示词模板
OPTIMIZED_PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 优化2: 实现简单的结果缓存机制
class SimpleCache:
    def __init__(self, max_size=100, ttl=3600):  # 默认缓存1小时
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['value']
            else:
                # 过期了，删除
                del self.cache[key]
        return None
    
    def set(self, key, value):
        # 如果缓存满了，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

class LLMProvider(LLMProviderBase):
    # 类级别缓存，所有实例共享
    _query_cache = SimpleCache()
    # 类级别的模型和向量库，避免重复加载
    _embedding_model = None
    _vectorstore = None
    
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

        # 优化3: 延迟加载和共享模型实例
        self._initialize_rag_components()

    def _initialize_rag_components(self):
        # 只在第一次调用时初始化
        if LLMProvider._embedding_model is None:
            start_time = time.time()
            model_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "models/bge-large-zh"
            
            # 优化4: 根据环境选择设备
            # 注意: 如果有GPU，可以将device改为"cuda"以加速嵌入生成
            device = "cpu"  # 如果有GPU可用，改为"cuda"
            
            LLMProvider._embedding_model = HuggingFaceEmbeddings(
                model_name=str(model_path),
                model_kwargs={"device": device}
            )
            logger.info(f"[RAG] Embedding 模型加载完成，耗时: {time.time() - start_time:.2f}秒")

        if LLMProvider._vectorstore is None:
            start_time = time.time()
            faiss_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "data/faiss_index_DeepSeek"
            if not faiss_path.exists():
                raise FileNotFoundError(f"未找到向量库: {faiss_path}")

            LLMProvider._vectorstore = FAISS.load_local(
                str(faiss_path),
                LLMProvider._embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"[RAG] 向量库加载完成，耗时: {time.time() - start_time:.2f}秒")

        # 优化5: 调整检索参数k值
        # 注意: k值是检索的文档数量，较小的k值可能会加快响应速度但可能影响答案质量
        # 建议根据实际情况测试不同的k值(1-5)找到最佳平衡点
        retriever_k = 2  # 从3减少到2，可以根据测试结果调整
        
        # 创建LLM实例
        llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            temperature=0.2,
        )
        
        # 创建检索器
        retriever = LLMProvider._vectorstore.as_retriever(search_kwargs={"k": retriever_k})
        
        # 创建QA链，使用优化的提示词模板
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": OPTIMIZED_PROMPT},
            return_source_documents=True,
        )
        
        # 保存检索器的引用，用于真正的流式响应
        self.retriever = retriever
        self.llm = llm

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

    # 优化6: 实现真正的流式RAG响应
    def rag_response_stream(self, query: str):
        logger.info(f"[RAG-STREAM] 开始流式处理查询: {query}")
        
        # 检查缓存
        cached_result = LLMProvider._query_cache.get(query)
        if cached_result:
            logger.info("[RAG-STREAM] 命中缓存，直接返回缓存结果")
            cleaned_text = self.clean_rag_text(cached_result)
            
            # 模拟流式返回缓存结果
            buffer = ""
            for sentence in re.split(r'(。|！|\!|\\?|\\？)', cleaned_text):
                if sentence.strip():
                    buffer += sentence
                    if len(buffer) >= 50:  # 减小缓冲区大小，更快返回第一个结果
                        yield buffer.strip(), None
                        buffer = ""
            
            if buffer.strip():
                yield buffer.strip(), None
            return
        
        try:
            # 步骤1: 先执行检索，获取相关文档
            start_time = time.time()
            relevant_docs = self.retriever.get_relevant_documents(query)
            retrieval_time = time.time() - start_time
            logger.info(f"[RAG-STREAM] 检索完成，耗时: {retrieval_time:.2f}秒，找到{len(relevant_docs)}个相关文档")
            
            # 步骤2: 构建提示词
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt_input = OPTIMIZED_PROMPT.format(context=context, question=query)
            
            # 步骤3: 流式调用LLM
            start_time = time.time()
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_input}],
                stream=True,
                temperature=0.2,
                max_tokens=self.max_tokens
            )
            
            # 步骤4: 流式返回结果
            full_response = ""
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content, None
            
            generation_time = time.time() - start_time
            logger.info(f"[RAG-STREAM] 生成完成，耗时: {generation_time:.2f}秒")
            
            # 缓存完整响应
            LLMProvider._query_cache.set(query, full_response)
            
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
