from flask import Flask, request, jsonify
from flasgger import Swagger
from app.predict import predict_answer

app = Flask(__name__)
Swagger(app)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict answer from context and question
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - question
            - context
          properties:
            question:
              type: string
              example: What is the capital of France?
            context:
              type: string
              example: Paris is the capital of France.
    responses:
      200:
        description: Successful response
        schema:
          type: object
          properties:
            answer:
              type: string
      400:
        description: Missing input
    """

    data = request.get_json()

    question = data.get("question")
    context = data.get("context")

    if not question or not context:
        return jsonify({"error": "Provide 'question' and 'context'"}), 400

    try:
        answer = predict_answer(question, context)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# debug helper (optional)
print("URL MAP:", app.url_map)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)