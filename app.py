from flask import Flask, request, render_template_string, jsonify
import requests, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "resonance-secret-key")

# ── Emotion Model ────────────────────────────────────────────
MODEL_PATH = "./emotion_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
emotion_classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
emotion_classifier.eval()

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

EMOTION_META = {
    "admiration":    {"color": "#7eb8f7", "emoji": "🌟"},
    "amusement":     {"color": "#f7c948", "emoji": "😄"},
    "anger":         {"color": "#e05c5c", "emoji": "😠"},
    "annoyance":     {"color": "#e07b5c", "emoji": "😤"},
    "approval":      {"color": "#6dd4a0", "emoji": "👍"},
    "caring":        {"color": "#a8d8ea", "emoji": "💙"},
    "confusion":     {"color": "#b0b3ff", "emoji": "🤔"},
    "curiosity":     {"color": "#9eceff", "emoji": "🔍"},
    "desire":        {"color": "#f7a8d8", "emoji": "✨"},
    "disappointment":{"color": "#8a9bb0", "emoji": "😞"},
    "disapproval":   {"color": "#c47a7a", "emoji": "👎"},
    "disgust":       {"color": "#a0704a", "emoji": "🤢"},
    "embarrassment": {"color": "#f7a8a8", "emoji": "😳"},
    "excitement":    {"color": "#ffb347", "emoji": "🎉"},
    "fear":          {"color": "#9b8fbf", "emoji": "😨"},
    "gratitude":     {"color": "#6ecf8c", "emoji": "🙏"},
    "grief":         {"color": "#7a8ca0", "emoji": "💔"},
    "joy":           {"color": "#f9e04b", "emoji": "😊"},
    "love":          {"color": "#f7a8c8", "emoji": "❤️"},
    "nervousness":   {"color": "#b8d4f7", "emoji": "😰"},
    "optimism":      {"color": "#8ed6a0", "emoji": "🌈"},
    "pride":         {"color": "#b89fff", "emoji": "🦁"},
    "realization":   {"color": "#f7d978", "emoji": "💡"},
    "relief":        {"color": "#a8e6c8", "emoji": "😌"},
    "remorse":       {"color": "#8fa0b8", "emoji": "😔"},
    "sadness":       {"color": "#7a9abf", "emoji": "😢"},
    "surprise":      {"color": "#f7c06e", "emoji": "😲"},
    "neutral":       {"color": "#a0a8b8", "emoji": "😐"},
}

# ── API Keys ─────────────────────────────────────────────────
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ── Models ───────────────────────────────────────────────────
# Groq — genuinely free, fast, reliable (PRIMARY)
GROQ_MODELS = [
    {"id": "llama-3.3-70b-versatile",  "label": "Llama 3.3 70B (Groq)",  "provider": "groq"},
    {"id": "llama3-70b-8192",          "label": "Llama 3 70B (Groq)",    "provider": "groq"},
    {"id": "mixtral-8x7b-32768",       "label": "Mixtral 8x7B (Groq)",   "provider": "groq"},
    {"id": "gemma2-9b-it",             "label": "Gemma 2 9B (Groq)",     "provider": "groq"},
    {"id": "llama-3.1-8b-instant",     "label": "Llama 3.1 8B (Groq)",   "provider": "groq"},
]

# OpenRouter — optional fallback
OPENROUTER_MODELS = [
    {"id": "deepseek/deepseek-chat-v3-0324:free",    "label": "DeepSeek V3 (OR)",  "provider": "openrouter"},
    {"id": "deepseek/deepseek-r1:free",              "label": "DeepSeek R1 (OR)",  "provider": "openrouter"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 (OR)",   "provider": "openrouter"},
]

ALL_MODELS = GROQ_MODELS + OPENROUTER_MODELS

# ── History ──────────────────────────────────────────────────
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
            migrated = []
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    migrated.append({"role": entry[0], "text": entry[1]})
                elif isinstance(entry, dict):
                    migrated.append(entry)
            return migrated
        except Exception:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

chat_history = load_history()

# ── Emotion Detection ─────────────────────────────────────────
def detect_emotion(text: str) -> tuple[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = emotion_classifier(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    idx = torch.argmax(probs, dim=-1).item()
    return EMOTION_LABELS[idx], round(probs[0][idx].item() * 100, 1)

# ── System Prompts ────────────────────────────────────────────
def get_system_prompt(emotion: str) -> str:
    if emotion in ["sadness", "grief", "remorse"]:
        return "You are a warm, gentle companion. The user is experiencing deep pain. Listen first, validate their feelings without minimizing them, then offer soft comfort. Be present, not prescriptive. Keep responses concise and human."
    elif emotion in ["disappointment", "disapproval", "disgust", "annoyance", "anger"]:
        return "You are a calm, grounded presence helping someone process frustration. Acknowledge their feelings, offer perspective without dismissing, and gently redirect toward calm. Be concise."
    elif emotion in ["fear", "nervousness"]:
        return "You are a steady, reassuring guide. Help the user feel safe and grounded. Use calming language and gentle reasoning. Be concise."
    elif emotion in ["joy", "excitement", "amusement", "love", "admiration", "pride"]:
        return "You are an enthusiastic, warm companion matching the user's positive energy. Celebrate with them genuinely! Be fun and engaging."
    elif emotion in ["gratitude", "relief", "optimism", "approval", "caring", "desire"]:
        return "You are a warm, appreciative companion. Acknowledge the user's positive feelings with sincerity and gentle encouragement."
    elif emotion in ["confusion", "curiosity", "realization", "surprise"]:
        return "You are a thoughtful intellectual companion. Help the user explore ideas and satisfy their curiosity with depth and nuance."
    return "You are a friendly, attentive assistant. Be natural, helpful, and engaging. Keep responses concise."

# ── LLM Providers ─────────────────────────────────────────────
def call_groq(system_prompt: str, messages: list, model_id: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("No Groq API key")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": 0.75,
            "max_tokens": 600,
        },
        timeout=20
    )
    data = resp.json()
    print(f"[Groq] model={model_id} status={resp.status_code}")
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    raise ValueError(data.get("error", {}).get("message", str(data)))

def call_openrouter(system_prompt: str, messages: list, model_id: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("No OpenRouter API key")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Resonance",
        },
        json={
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": 0.75,
            "max_tokens": 600,
        },
        timeout=30
    )
    data = resp.json()
    print(f"[OpenRouter] model={model_id} status={resp.status_code}")
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    raise ValueError(data.get("error", {}).get("message", str(data)))

def call_llm(system_prompt: str, user_message: str, history: list, preferred_model_id: str = None) -> tuple[str, str]:
    """Returns (reply_text, model_label_used)"""
    recent = history[-12:] if len(history) > 12 else history
    messages = []
    for entry in recent:
        role = "user" if entry["role"] == "user" else "assistant"
        messages.append({"role": role, "content": entry["text"]})
    messages.append({"role": "user", "content": user_message})

    # Build candidate list: preferred first, then all others
    if preferred_model_id:
        preferred = next((m for m in ALL_MODELS if m["id"] == preferred_model_id), None)
        others    = [m for m in ALL_MODELS if m["id"] != preferred_model_id]
        candidates = ([preferred] if preferred else []) + others
    else:
        candidates = ALL_MODELS

    last_error = "Unknown error"
    for model in candidates:
        try:
            if model["provider"] == "groq":
                reply = call_groq(system_prompt, messages, model["id"])
            else:
                reply = call_openrouter(system_prompt, messages, model["id"])
            return reply, model["label"]
        except Exception as e:
            last_error = str(e)
            print(f"[LLM] SKIP {model['id']}: {last_error}")
            continue

    return (
        "⚠️ Could not get a response.\n\n"
        "Make sure GROQ_API_KEY is set in your .env file.\n"
        "Get a free key at: console.groq.com",
        "none"
    )

# ── Routes ────────────────────────────────────────────────────
@app.route("/models")
def get_models():
    visible = []
    for m in ALL_MODELS:
        if m["provider"] == "groq" and GROQ_API_KEY:
            visible.append(m)
        elif m["provider"] == "openrouter" and OPENROUTER_API_KEY:
            visible.append(m)
    if not visible:
        visible = [{"id": "none", "label": "⚠️ No API key — add GROQ_API_KEY to .env", "provider": "none"}]
    return jsonify(visible)

@app.route("/")
def chat():
    return render_template_string(TEMPLATE, chat_history=chat_history)

@app.route("/send", methods=["POST"])
def send():
    global chat_history
    data = request.get_json()
    user_input = (data.get("message") or "").strip()
    model_id   = data.get("model") or None
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    emotion, confidence = detect_emotion(user_input)
    system_prompt = get_system_prompt(emotion)
    bot_reply, model_used = call_llm(system_prompt, user_input, chat_history, model_id)

    meta      = EMOTION_META.get(emotion, {"color": "#a0a8b8", "emoji": "😐"})
    timestamp = datetime.now().strftime("%H:%M")

    chat_history.append({"role": "user", "text": user_input, "time": timestamp})
    chat_history.append({
        "role": "bot", "text": bot_reply,
        "emotion": emotion, "confidence": confidence,
        "emoji": meta["emoji"], "color": meta["color"],
        "time": timestamp, "model": model_used,
    })
    save_history(chat_history)

    return jsonify({
        "bot_reply": bot_reply,
        "emotion": emotion, "confidence": confidence,
        "emoji": meta["emoji"], "color": meta["color"],
        "time": timestamp, "model_used": model_used,
    })

@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history = []
    save_history(chat_history)
    return jsonify({"status": "cleared"})

# ── Template ──────────────────────────────────────────────────
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Resonance — Emotion-Aware Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg: #0f1117; --surface: #171b26; --surface2: #1e2333; --border: #2a3045;
  --text: #e8eaf2; --text-muted: #6b7394; --text-subtle: #3d4460;
  --accent: #7eb8f7; --accent-glow: rgba(126,184,247,0.15);
  --user-bg: #1a2640; --user-border: #2a4070; --bot-bg: #1a1f30;
  --radius: 18px;
  --font-display: 'Instrument Serif', Georgia, serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
  --transition: 0.25s cubic-bezier(0.4,0,0.2,1);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: var(--font-body); background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }
body::before {
  content: ''; position: fixed; inset: 0;
  background: radial-gradient(ellipse 60% 50% at 20% 10%, rgba(126,184,247,0.04) 0%, transparent 60%),
              radial-gradient(ellipse 50% 60% at 80% 90%, rgba(180,120,247,0.04) 0%, transparent 60%);
  pointer-events: none; z-index: 0;
}
.layout { display: flex; flex-direction: column; height: 100vh; max-width: 820px; margin: 0 auto; position: relative; z-index: 1; }
.header { display: flex; align-items: center; justify-content: space-between; padding: 18px 28px 14px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.header-left { display: flex; align-items: baseline; gap: 10px; }
.logo { font-family: var(--font-display); font-size: 26px; color: var(--text); letter-spacing: -0.5px; font-style: italic; }
.logo-sub { font-size: 11px; color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase; font-weight: 400; }
.header-actions { display: flex; gap: 8px; align-items: center; }
.model-select-wrap select {
  background: var(--surface); border: 1px solid var(--border); color: var(--text-muted);
  font-family: var(--font-body); font-size: 12px; padding: 7px 10px; border-radius: 10px;
  cursor: pointer; outline: none; transition: var(--transition); max-width: 210px;
}
.model-select-wrap select:hover, .model-select-wrap select:focus { border-color: var(--accent); color: var(--text); }
.icon-btn {
  background: none; border: 1px solid var(--border); color: var(--text-muted); cursor: pointer;
  border-radius: 10px; padding: 7px 12px; font-size: 13px; font-family: var(--font-body);
  transition: var(--transition); display: flex; align-items: center; gap: 6px;
}
.icon-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }
.emotion-bar { height: 3px; background: var(--border); flex-shrink: 0; overflow: hidden; }
.emotion-bar-fill { height: 100%; width: 0%; transition: width 0.8s cubic-bezier(0.4,0,0.2,1), background-color 0.6s ease; border-radius: 2px; }
.messages { flex: 1; overflow-y: auto; padding: 24px 28px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
.messages::-webkit-scrollbar { width: 4px; }
.messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; color: var(--text-subtle); text-align: center; pointer-events: none; padding: 40px; }
.empty-icon { font-size: 48px; opacity: 0.4; }
.empty-title { font-family: var(--font-display); font-size: 22px; font-style: italic; color: var(--text-muted); }
.empty-sub { font-size: 13px; line-height: 1.6; max-width: 280px; }
.message-row { display: flex; flex-direction: column; animation: slideUp 0.35s cubic-bezier(0.4,0,0.2,1) both; }
.message-row.user { align-items: flex-end; }
.message-row.bot  { align-items: flex-start; }
@keyframes slideUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
.bubble { padding: 14px 18px; border-radius: var(--radius); max-width: 78%; line-height: 1.65; font-size: 15px; font-weight: 300; white-space: pre-wrap; word-break: break-word; }
.bubble.user { background: var(--user-bg); border: 1px solid var(--user-border); border-bottom-right-radius: 5px; }
.bubble.bot  { background: var(--bot-bg); border: 1px solid var(--border); border-bottom-left-radius: 5px; }
.bubble-meta { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 11px; color: var(--text-subtle); flex-wrap: wrap; }
.message-row.user .bubble-meta { flex-direction: row-reverse; }
.emotion-tag { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 20px; font-size: 10.5px; font-weight: 500; letter-spacing: 0.04em; text-transform: lowercase; border: 1px solid; }
.model-tag { font-size: 10px; opacity: 0.5; }
.typing-row { display: flex; align-items: flex-start; animation: slideUp 0.3s ease both; }
.typing-bubble { background: var(--bot-bg); border: 1px solid var(--border); border-radius: var(--radius); border-bottom-left-radius: 5px; padding: 14px 20px; display: flex; gap: 6px; align-items: center; }
.dot { width: 7px; height: 7px; background: var(--text-subtle); border-radius: 50%; animation: bounce 1.2s ease infinite; }
.dot:nth-child(2) { animation-delay: 0.2s; } .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100% { transform: scale(0.7); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
.input-area { padding: 16px 28px 24px; border-top: 1px solid var(--border); flex-shrink: 0; }
.input-row { display: flex; align-items: flex-end; gap: 10px; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 10px 10px 10px 18px; transition: border-color var(--transition), box-shadow var(--transition); }
.input-row:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
textarea { flex: 1; background: none; border: none; outline: none; color: var(--text); font-family: var(--font-body); font-size: 15px; font-weight: 300; line-height: 1.5; resize: none; max-height: 140px; overflow-y: auto; padding: 4px 0; min-height: 26px; }
textarea::placeholder { color: var(--text-subtle); }
.send-btn { background: var(--accent); border: none; color: #0a0e1a; cursor: pointer; border-radius: 11px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: var(--transition); }
.send-btn:hover { background: #a8d0ff; transform: scale(1.05); }
.send-btn:active { transform: scale(0.96); }
.send-btn:disabled { opacity: 0.3; cursor: default; transform: none; }
.input-hint { font-size: 11px; color: var(--text-subtle); text-align: center; margin-top: 10px; }
.toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%) translateY(20px); background: var(--surface2); border: 1px solid var(--border); color: var(--text-muted); padding: 10px 20px; border-radius: 10px; font-size: 13px; opacity: 0; pointer-events: none; transition: all 0.3s ease; z-index: 100; }
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
@media (max-width: 600px) { .header, .messages, .input-area { padding-left: 16px; padding-right: 16px; } .bubble { max-width: 90%; } .logo { font-size: 22px; } }
</style>
</head>
<body>
<div class="layout">
  <header class="header">
    <div class="header-left">
      <span class="logo">Resonance</span>
      <span class="logo-sub">emotion-aware</span>
    </div>
    <div class="header-actions">
      <div class="model-select-wrap">
        <select id="modelSelect"><option value="">Loading…</option></select>
      </div>
      <button class="icon-btn" id="clearBtn">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
        Clear
      </button>
    </div>
  </header>
  <div class="emotion-bar"><div class="emotion-bar-fill" id="emotionFill"></div></div>
  <div class="messages" id="messages">
    {% if not chat_history %}
    <div class="empty-state" id="emptyState">
      <div class="empty-icon">✦</div>
      <div class="empty-title">How are you feeling?</div>
      <div class="empty-sub">I'll listen carefully and respond to what you're actually experiencing right now.</div>
    </div>
    {% else %}
      {% for entry in chat_history %}
        {% if entry.role == 'user' %}
        <div class="message-row user">
          <div class="bubble user">{{ entry.text }}</div>
          <div class="bubble-meta"><span>{{ entry.get('time', '') }}</span></div>
        </div>
        {% else %}
        <div class="message-row bot">
          <div class="bubble bot">{{ entry.text }}</div>
          <div class="bubble-meta">
            {% if entry.get('emotion') %}
            <span class="emotion-tag" style="color:{{ entry.color }};border-color:{{ entry.color }}30;background:{{ entry.color }}10;">
              {{ entry.emoji }} {{ entry.emotion }} · {{ entry.confidence }}%
            </span>
            {% endif %}
            <span>{{ entry.get('time', '') }}</span>
            {% if entry.get('model') %}<span class="model-tag">· {{ entry.model }}</span>{% endif %}
          </div>
        </div>
        {% endif %}
      {% endfor %}
    {% endif %}
  </div>
  <div class="input-area">
    <div class="input-row">
      <textarea id="input" placeholder="Share what's on your mind…" rows="1" autocomplete="off"></textarea>
      <button class="send-btn" id="sendBtn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
      </button>
    </div>
    <div class="input-hint">Enter to send &nbsp;·&nbsp; Shift+Enter for new line</div>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
const messagesEl=document.getElementById('messages'),inputEl=document.getElementById('input'),
      sendBtn=document.getElementById('sendBtn'),emotionFill=document.getElementById('emotionFill'),
      clearBtn=document.getElementById('clearBtn'),toastEl=document.getElementById('toast'),
      modelSelect=document.getElementById('modelSelect');

fetch('/models').then(r=>r.json()).then(models=>{
  modelSelect.innerHTML='';
  models.forEach((m,i)=>{
    const o=document.createElement('option');
    o.value=m.id; o.textContent=m.label; if(i===0)o.selected=true;
    modelSelect.appendChild(o);
  });
}).catch(()=>{ modelSelect.innerHTML='<option value="">Default</option>'; });

const showToast=msg=>{ toastEl.textContent=msg; toastEl.classList.add('show'); setTimeout(()=>toastEl.classList.remove('show'),2400); };
const scrollBottom=()=>messagesEl.scrollTo({top:messagesEl.scrollHeight,behavior:'smooth'});
const removeEmpty=()=>{ const e=document.getElementById('emptyState'); if(e)e.remove(); };
const escHtml=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');

function appendUserBubble(text,time){
  const row=document.createElement('div'); row.className='message-row user';
  row.innerHTML=`<div class="bubble user">${escHtml(text)}</div><div class="bubble-meta"><span>${time}</span></div>`;
  messagesEl.appendChild(row); scrollBottom();
}
function appendTyping(){
  const row=document.createElement('div'); row.className='typing-row'; row.id='typing';
  row.innerHTML=`<div class="typing-bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  messagesEl.appendChild(row); scrollBottom();
}
function removeTyping(){ const t=document.getElementById('typing'); if(t)t.remove(); }

function appendBotBubble(text,emotion,confidence,emoji,color,time,modelUsed){
  const row=document.createElement('div'); row.className='message-row bot';
  const eTag=emotion?`<span class="emotion-tag" style="color:${color};border-color:${color}30;background:${color}10;">${emoji} ${emotion} · ${confidence}%</span>`:'';
  const mTag=(modelUsed&&modelUsed!=='none')?`<span class="model-tag">· ${modelUsed}</span>`:'';
  row.innerHTML=`<div class="bubble bot">${escHtml(text)}</div><div class="bubble-meta">${eTag}<span>${time}</span>${mTag}</div>`;
  messagesEl.appendChild(row); scrollBottom();
  if(color){ emotionFill.style.background=color; emotionFill.style.width=`${Math.min(confidence,99)}%`; }
}

async function sendMessage(){
  const text=inputEl.value.trim(); if(!text)return;
  removeEmpty();
  const time=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  appendUserBubble(text,time);
  inputEl.value=''; inputEl.style.height='auto'; sendBtn.disabled=true;
  appendTyping();
  try{
    const res=await fetch('/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,model:modelSelect.value})});
    const data=await res.json(); removeTyping();
    if(data.error) appendBotBubble('⚠️ '+data.error,null,0,'','',time,null);
    else appendBotBubble(data.bot_reply,data.emotion,data.confidence,data.emoji,data.color,data.time,data.model_used);
  }catch(e){ removeTyping(); appendBotBubble('⚠️ Network error. Please try again.',null,0,'','',time,null); }
  sendBtn.disabled=false; inputEl.focus();
}

inputEl.addEventListener('input',function(){ this.style.height='auto'; this.style.height=Math.min(this.scrollHeight,140)+'px'; });
inputEl.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMessage(); } });
sendBtn.addEventListener('click',sendMessage);
clearBtn.addEventListener('click',async()=>{
  if(!confirm('Clear the entire conversation?'))return;
  await fetch('/clear',{method:'POST'});
  messagesEl.innerHTML=`<div class="empty-state" id="emptyState"><div class="empty-icon">✦</div><div class="empty-title">How are you feeling?</div><div class="empty-sub">I'll listen carefully and respond to what you're actually experiencing right now.</div></div>`;
  emotionFill.style.width='0%'; showToast('Conversation cleared');
});
scrollBottom(); inputEl.focus();
</script>
</body>
</html>"""

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    print(f"\n{'='*55}")
    print(f"  Groq key:       {'✅ SET' if GROQ_API_KEY else '❌ MISSING — get free at console.groq.com'}")
    print(f"  OpenRouter key: {'✅ SET (fallback)' if OPENROUTER_API_KEY else '⚠️  not set (optional)'}")
    print(f"{'='*55}\n")
    app.run(debug=debug, port=port)