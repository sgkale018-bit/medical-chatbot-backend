from flask import Flask, request, jsonify

app = Flask(__name__)

# Smart response function
def get_ai_response(message):
    message = message.lower()

    if "fever" in message:
        return "You may have a viral infection. Stay hydrated and consult a doctor if symptoms persist."
    elif "headache" in message:
        return "Headaches can be caused by stress, dehydration, or lack of sleep."
    elif "cough" in message:
        return "A cough may indicate a cold or respiratory issue. Monitor your symptoms."
    elif "stomach" in message:
        return "Stomach pain could be due to indigestion or infection. Consider consulting a doctor."
    else:
        return "Please consult a medical professional for accurate advice."

# Home route
@app.route("/")
def home():
    return "Medical Chatbot Backend Running"

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")

    response = get_ai_response(message)

    return jsonify({"answer": response})

# Run server
if __name__ == "__main__":
    app.run(debug=True)