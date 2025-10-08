import os
import requests
import json
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_URL_TTS = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key="
API_URL_LLM = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="

try:
    model = SentenceTransformer("fine_tuned_chatbot_model")
    with open("Chat_Data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item['prompt'] for item in data]
    answers = [item['response'] for item in data]
    
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    print("Chatbot model and data loaded successfully!")
    model_loaded = True
except Exception as e:
    print(f"Error loading model or data: {e}")
    model_loaded = False

def llm_fallback_response(user_input):
    """Generates a creative response using Gemini API when a match is not found."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": user_input}]}]
    }
    try:
        full_api_url = API_URL_LLM + os.environ.get("GOOGLE_API_KEY", "")
        response = requests.post(full_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Gemini LLM API request failed: {e}")
        return "I'm sorry, I couldn't find a good answer for that."

def chatbot_response(user_input):
    if not model_loaded:
        return "Sorry, the chatbot model failed to load. Please check the files."
        
    user_emb = model.encode(user_input, convert_to_tensor=True)
    hits = util.semantic_search(user_emb, question_embeddings, top_k=1)
    best_match = hits[0][0]
    
    if best_match['score'] > 0.5:
        best_idx = best_match['corpus_id']
        return answers[best_idx]
    else:
        print(f"No strong match found. Falling back to LLM for: '{user_input}'")
        return llm_fallback_response(user_input)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    if user_message:
        bot_response = chatbot_response(user_message)
        return jsonify({"response": bot_response})
    return jsonify({"response": "I didn't receive a message."}), 400

@app.route("/tts", methods=["POST"])
def tts():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Kore" } }
            }
        }
    }

    try:
        full_api_url = API_URL_TTS + os.environ.get("GOOGLE_API_KEY", "")
        response = requests.post(full_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        audio_data_base64 = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
        
        audio_data_bytes = base64.b64decode(audio_data_base64)
        
        raw_audio = AudioSegment(
            data=audio_data_bytes,
            sample_width=2, 
            frame_rate=16000,
            channels=1
        )

        wav_buffer = io.BytesIO()
        raw_audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        return send_file(wav_buffer, mimetype="audio/wav")

    except requests.exceptions.RequestException as e:
        print(f"Gemini TTS API request failed: {e}")
        return jsonify({"error": "Failed to get response from TTS service."}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
