import os
from flask import Flask, request, jsonify

import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

import faiss
import numpy as np
import json

app = Flask(__name__)

# ======================================================================
# ⚡ Vertex AI 配置
# ======================================================================
PROJECT_ID = "colab-20250607"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash"
EMBED_MODEL = "text-embedding-005"

# 预初始化服务账号
credentials = service_account.Credentials.from_service_account_file(
    "/secrets/vertex-json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials
)

embedder = GenerativeModel(EMBED_MODEL)
generator = GenerativeModel(MODEL_NAME)

# ======================================================================
# 数据路径
# ======================================================================
DOC_PATH = "knowledge.json"
EMBED_PATH = "embedding.npy"
INDEX_PATH = "faiss_index.index"

# ======================================================================
# 加载知识库
# ======================================================================
def load_knowledge():
    if not os.path.exists(DOC_PATH):
        print("❗ knowledge.json 不存在")
        return []
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

DOCUMENTS = load_knowledge()

# ======================================================================
# Vertex AI Embedding 函数（替代 HuggingFace）
# ======================================================================
def embed_texts(text_list):
    """使用 Vertex AI text-embedding-005 生成向量"""
    responses = embedder.generate_content(text_list)
    # responses.embeddings 是 list[Embedding]
    vecs = np.array([e.values for e in responses.embeddings])
    return vecs

# ======================================================================
# 构建索引
# ======================================================================
def build_index(documents):
    print("⚡ 生成文档向量（Vertex AI Embedding）...")
    embeddings = embed_texts(documents)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    np.save(EMBED_PATH, embeddings)

    print("✅ 向量索引已生成")
    return index


def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBED_PATH):
        print("📥 加载已有索引")
        return faiss.read_index(INDEX_PATH)
    else:
        print("⚠ 没有索引，正在构建")
        return build_index(DOCUMENTS)

INDEX = load_index()

# ======================================================================
# Prompt 模板
# ======================================================================
def build_prompt(context, query):
    return f"""
你是一位专业的 Vertex AI 助手。请根据以下资料回答问题：

【相关资料】
{context}

【用户问题】
{query}

请给出准确、清晰、基于资料的回答：
"""

# ======================================================================
# /generate 主接口（Cloud Run 调用）
# ======================================================================
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "缺少 query 字段"}), 400

    # ------------------------------------------------------------
    # 1）向量召回（RAG）
    # ------------------------------------------------------------
    query_vec = embed_texts([query])

    D, I = INDEX.search(np.array(query_vec), k=3)
    relevant_docs = [DOCUMENTS[i] for i in I[0] if i < len(DOCUMENTS)]

    context = "\n".join(relevant_docs)

    # ------------------------------------------------------------
    # 2）构造 Prompt
    # ------------------------------------------------------------
    prompt = build_prompt(context, query)

    # ------------------------------------------------------------
    # 3）调用 Gemini 生成最终回答
    # ------------------------------------------------------------
    responses = generator.generate_content(prompt, stream=True)
    result = "".join([chunk.text for chunk in responses])

    return jsonify({"result": result})

# ======================================================================
# Cloud Run 必须监听 PORT
# ======================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Flask running on port {port}")
    app.run(host="0.0.0.0", port=port)
