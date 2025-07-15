import sys
import types
import pytest

sys.path.append('main/xiaozhi-server')

dummy_names = [
    'langchain_community.vectorstores',
    'langchain_huggingface',
    'langchain.chains',
    'langchain_openai',
    'langchain.prompts',
    'sentence_transformers',
    'mysql.connector',
    'core.utils.util',
    'config.logger',
    'config.config_loader',
    'config.settings',
    'config.manage_api_client',
    'requests',
]
for name in dummy_names:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

sys.modules['langchain_community.vectorstores'].FAISS = types.SimpleNamespace(load_local=lambda *a, **k: None)
sys.modules['langchain_huggingface'].HuggingFaceEmbeddings = object
sys.modules['langchain.chains'].RetrievalQA = object
sys.modules['langchain_openai'].ChatOpenAI = object
sys.modules['langchain.prompts'].PromptTemplate = types.SimpleNamespace(from_template=lambda t: t)
sys.modules['sentence_transformers'].CrossEncoder = object
sys.modules['mysql.connector'].connect = lambda *a, **k: None
sys.modules['mysql.connector'].Error = Exception
sys.modules['mysql'] = types.ModuleType('mysql')
sys.modules['mysql'].connector = sys.modules['mysql.connector']
sys.modules['core.utils.util'].check_model_key = lambda *a, **k: None
class _DummyLogger:
    def info(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass
    def bind(self, **kwargs):
        return self

sys.modules['config.logger'].setup_logging = lambda: _DummyLogger()
sys.modules['config.config_loader'].load_config = lambda *a, **k: {}
sys.modules['config.settings'].check_config_file = lambda *a, **k: None
sys.modules['config.manage_api_client'].init_service = lambda *a, **k: None
sys.modules['config.manage_api_client'].get_server_config = lambda *a, **k: {}
sys.modules['config.manage_api_client'].get_agent_models = lambda *a, **k: {}

import core.providers.llm.openai.openai as openai_module
from core.providers.llm.openai.openai import LLMProvider

class DummyStream:
    def __init__(self, text):
        self._text = text
    def __iter__(self):
        for ch in self._text:
            chunk = types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=ch))])
            yield chunk

class DummyResp:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]

@pytest.fixture(autouse=True)
def patch_init(monkeypatch):
    monkeypatch.setattr(LLMProvider, "_initialize_rag_components", lambda self: None)


def build_provider(monkeypatch, recorder=None):
    provider = LLMProvider({"model_name": "gpt", "api_key": "key", "url": "http://"})
    provider.retriever = types.SimpleNamespace(get_relevant_documents=lambda q: [])
    LLMProvider._query_cache = openai_module.SimpleCache()
    if recorder is None:
        monkeypatch.setattr(provider, "save_text_to_mysql", lambda *a, **k: None)
    else:
        monkeypatch.setattr(provider, "save_text_to_mysql", recorder)
    return provider


def test_extract_keywords(monkeypatch):
    provider = build_provider(monkeypatch)
    provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: DummyResp("流血,恶心"))))
    assert provider.extract_keywords("为什么会流血并恶心") == "流血,恶心"


def test_rerank_docs(monkeypatch):
    provider = build_provider(monkeypatch)
    class FakeCE:
        def predict(self, pairs):
            return [len(p[1]) for p in pairs]
    monkeypatch.setattr(LLMProvider, "_cross_encoder", FakeCE())
    docs = [types.SimpleNamespace(page_content="a"), types.SimpleNamespace(page_content="longer")]
    ranked = provider.rerank_docs("q", docs)
    assert ranked[0].page_content == "longer"


def test_rag_response_stream_with_docs(monkeypatch):
    provider = build_provider(monkeypatch)
    monkeypatch.setattr(provider, "extract_keywords", lambda q: "kw")
    provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: DummyStream("答案"))))
    provider.retriever = types.SimpleNamespace(get_relevant_documents=lambda q: [types.SimpleNamespace(page_content="doc")])
    monkeypatch.setattr(provider, "rerank_docs", lambda q, d: d)
    out = ''.join([o[0] for o in provider.rag_response_stream("问题")])
    assert "答" in out


def test_rag_response_stream_no_docs(monkeypatch):
    provider = build_provider(monkeypatch)
    monkeypatch.setattr(provider, "extract_keywords", lambda q: "kw")
    provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: DummyStream("普通"))))
    provider.retriever = types.SimpleNamespace(get_relevant_documents=lambda q: [])
    monkeypatch.setattr(provider, "rerank_docs", lambda q, d: d)
    out = ''.join([o[0] for o in provider.rag_response_stream("问题")])
    assert "普" in out


def test_mysql_logging(monkeypatch):
    records = []
    def recorder(mac, types, content):
        records.append((types, content))

    provider = build_provider(monkeypatch, recorder)
    monkeypatch.setattr(provider, "extract_keywords", lambda q: (recorder('', 'keywords', 'kw') or 'kw'))
    provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: DummyStream("OK"))))
    provider.retriever = types.SimpleNamespace(get_relevant_documents=lambda q: [])
    monkeypatch.setattr(provider, "rerank_docs", lambda q, d: d)
    list(provider.rag_response_stream("问题"))
    logged_types = {t for t, _ in records}
    assert "keywords" in logged_types and "res" in logged_types
