from huggingface_hub import snapshot_download

# 将模型下载到指定目录（如 ./models/bge-reranker-v2-m3）
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="./models/bge-reranker-v2-m3",
)