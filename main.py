import os
from flask import Flask, request, jsonify
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerationConfig
from google.oauth2 import service_account
import vertexai

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# === 全局变量 ===
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DOC_PATH = "knowledge.json"
EMBED_PATH = "embedding.npy"
INDEX_PATH = "faiss_index.index"

def load_knowledge():
    if not os.path.exists(DOC_PATH):
        print("❗️未找到 knowledge.json 文件")
        return []
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(documents):
    embeddings = EMBEDDING_MODEL.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    np.save(EMBED_PATH, embeddings)
    return index

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBED_PATH):
        return faiss.read_index(INDEX_PATH)
    else:
        docs = load_knowledge()
        return build_index(docs)

DOCUMENTS = load_knowledge()
INDEX = load_index()

def build_prompt(context, query):
    return f"""你是一位专业的 Vertex AI 助手。以下是相关资料片段：

{context}

用户问题：{query}

请基于以上内容专业、准确地回答："""

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    query = data.get("query", "")

    # 向量检索
    query_embedding = EMBEDDING_MODEL.encode([query])
    D, I = INDEX.search(np.array(query_embedding), k=3)
    relevant_docs = [DOCUMENTS[i] for i in I[0] if i < len(DOCUMENTS)]
    context = "\n".join(relevant_docs)

    # Prompt 构造
    prompt = build_prompt(context, query)

    # Vertex AI 初始化
    credentials = service_account.Credentials.from_service_account_file(
        "/secrets/vertex-json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    vertexai.init(
        project="colab-20250607",
        location="us-central1",
        credentials=credentials
    )

    model = GenerativeModel("gemini-2.0-flash")
    config = GenerationConfig(temperature=0.7, top_p=1.0, max_output_tokens=1024)
    responses = model.generate_content(prompt, generation_config=config, stream=True)
    result = "".join([chunk.text for chunk in responses])

    return jsonify({"result": result})

# ✅ Cloud Run 要求监听 $PORT 环境变量
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"✅ Flask server starting on port {port}...")
    app.run(host="0.0.0.0", port=port)
