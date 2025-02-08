import os
import json
from flask import Flask, render_template, request, jsonify
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API"))
model = SentenceTransformer('all-MiniLM-L6-v2')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    """Handles audio upload and transcription"""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, "recorded_audio.wav")
    audio_file.save(file_path)

    # Transcribe using Whisper
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=("recorded_audio.wav", file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    
    transcribed_text = transcription.text.lower().strip()
    target_vector = model.encode(transcribed_text).reshape(1, -1)
    input_vector = model.encode("make flashcards").reshape(1, -1)

    # Compute cosine similarity
    similarity_score = cosine_similarity(target_vector, input_vector)[0][0]

    threshold = 0.7  
    
    # Check if command is "make flashcards"
    if similarity_score > threshold:
        return generate_flashcards()

    return jsonify({"transcription": transcribed_text})

def generate_flashcards():
    """Fetch text from frontend, generate Q&A using Groq"""
    text = request.form.get("text")  # Get the text from frontend
    if not text:
        return jsonify({"error": "No text found for flashcard generation."}), 400

    num_q = 5  # Adjust as needed
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an assistant for generating Q&A from text. Only return JSON. No extra text."},
            {"role": "user", "content": f'Generate {num_q} questions and answers based on the following text:\n\n"{text}" in JSON array format like: [{{"question": "...", "answer": "..."}}]'},
        ],
        temperature=0.7,
    )

    qa_content = completion.choices[0].message.content.strip()

    try:
        qa_pairs = json.loads(qa_content)  # Parse JSON response
    except json.JSONDecodeError:
        return jsonify({"error": f"Invalid JSON response: {qa_content}"}), 500

    return jsonify({"success": True, "flashcards": qa_pairs})

if __name__ == "__main__":
    app.run(debug=True)