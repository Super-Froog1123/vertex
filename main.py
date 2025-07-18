from flask import Flask, request, jsonify
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerationConfig
from google.oauth2 import service_account
import vertexai

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    query = data.get("query", "")

    credentials = service_account.Credentials.from_service_account_file(
        "/secrets/vertex/2.json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    vertexai.init(
        project="colab-20250607",
        location="us-central1",
        credentials=credentials
    )

    model = GenerativeModel("gemini-2.0-flash")
    config = GenerationConfig(temperature=1.0, top_p=1.0, max_output_tokens=1024)
    responses = model.generate_content(query, generation_config=config, stream=True)

    result = "".join([chunk.text for chunk in responses])
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
