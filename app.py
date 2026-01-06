from flask import Flask, request, render_template_string, redirect
import requests, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# === Load Emotion Model ===
MODEL_PATH = "./emotion_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

HISTORY_FILE = "history.json"

# === Load or Init Chat History ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

chat_history = load_history()

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    return emotion_labels[predicted_class]

def get_prompt(emotion):
    if emotion in ["sadness", "grief", "remorse", "disappointment"]:
        return "You are a gentle and compassionate assistant comforting someone feeling down."
    elif emotion in ["anger", "annoyance", "disapproval", "disgust"]:
        return "You are a calm and understanding assistant helping someone process frustration."
    elif emotion in ["fear", "nervousness"]:
        return "You are a reassuring assistant helping someone feel safe and grounded."
    elif emotion in ["joy", "amusement", "excitement", "love", "admiration", "pride", "relief", "gratitude", "optimism"]:
        return "You are an enthusiastic assistant celebrating joyful feelings with the user."
    elif emotion in ["confusion", "curiosity", "realization"]:
        return "You are a thoughtful assistant helping the user reflect and explore ideas."
    elif emotion == "neutral":
        return "You are a neutral, attentive assistant here to support the user with anything on their mind."
    else:
        return "You are an empathetic assistant who responds appropriately to the user's emotional state."

@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history
    if request.method == "POST":
        user_input = request.form["message"]
        emotion = detect_emotion(user_input)
        system_prompt = get_prompt(emotion)

        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "EmotionAwareBot"
        }

        payload = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }

        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                     headers=headers,
                                     data=json.dumps(payload))
            data = response.json()
            if "choices" in data:
                bot_reply = data["choices"][0]["message"]["content"]
            else:
                bot_reply = f"⚠️ API Error: {data.get('error', 'Unknown error')}"
        except Exception as e:
            bot_reply = f"⚠️ Error: {str(e)}"

        chat_history.append(("user", user_input))
        chat_history.append(("bot", bot_reply))
        save_history(chat_history)

        return redirect("/")

    return render_template_string(TEMPLATE, chat_history=chat_history)


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Emotion-Aware Chatbot</title>
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background: linear-gradient(to right, #f9f9f9, #e9ecf3);
        transition: background 0.4s, color 0.4s;
    }

    .wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }

.chat-wrapper {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 900px;
    height: 95vh; /* or use 100% if wrapper is 100% */
    background: white;
    border-radius: 14px;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}


.chat-messages {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 20px;
    overflow-y: auto;
}



.message {
    display: block; /* forces each message to be a full-width flex item */
    clear: both;    /* prevent float-type layout bleed */
    padding: 10px 15px;
    border-radius: 12px;
    margin: 10px 0;
    max-width: 75%;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    animation: fadeIn 0.4s ease-in-out;
}


.user {
    background: #007bff;
    color: white;
    align-self: flex-end;
    text-align: left;
    border-radius: 12px 12px 0 12px; /* Optional: make user bubbles look different */
}


.bot {
    background: #e6e6e6;
    color: black;
    align-self: flex-start;
    text-align: left;
    border-radius: 12px 12px 12px 0;
}


    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .input-box {
        display: flex;
        padding: 15px;
        border-top: 1px solid #ddd;
    }

    textarea {
        flex: 1;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 10px;
        font-size: 16px;
        resize: none;
    }

    button {
        margin-left: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        background-color: #007bff;
        color: white;
        cursor: pointer;
    }

    .toggle-btn {
        position: fixed;
        top: 10px;
        right: 20px;
        padding: 6px 12px;
        border-radius: 10px;
        z-index: 1000;
    }

    body.dark {
        background: linear-gradient(to right, #1f1f1f, #2a2a2a);
        color: #eee;
    }

    .dark .chat-wrapper {
        background: #1c1c1c;
    }

    .dark .message.bot {
        background: #2a2a2a;
        color: #eee;
    }

    .dark .message.user {
        background: #0d6efd;
    }

    .dark textarea {
        background: #1e1e1e;
        color: white;
        border: 1px solid #555;
    }
</style>
</head>
<body class="light">
    <button class="toggle-btn" onclick="toggleTheme()">🌗 Theme</button>
    <div class="wrapper">
        <div class="chat-wrapper">
            <div class="chat-messages" id="chat">
                {% for role, msg in chat_history %}
                    <div class="message {{ role }}">{{ msg | safe }}</div>
                {% endfor %}
            </div>
            <form class="input-box" method="POST">
                <textarea id="msg" name="message" placeholder="Type a message..." rows="1"></textarea>
                <button type="submit">Send</button>
            </form>
        </div>
    </div>



<script>
function toggleTheme() {
    document.body.classList.toggle("dark");
    document.body.classList.toggle("light");
}

document.getElementById("msg").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.form.submit();
    }
});

window.onload = () => {
    const chat = document.getElementById("chat");
    chat.scrollTop = chat.scrollHeight;
};
</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
