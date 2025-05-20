import os
import threading
import speech_recognition as sr
from threading import Thread
from dotenv import load_dotenv
from flask import Flask, send_from_directory
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import google.generativeai as genai

# ========== ENV + PAGE CONFIG ==========
load_dotenv()
st.set_page_config(page_title="Dr. HeAlify", page_icon="🧑‍⚕️", layout="centered", initial_sidebar_state="collapsed")

# ========== THEME SWITCHER ==========
theme_mode = st.toggle("🌙 Enable Dark Mode")

if theme_mode:
    bg_color = "#121212"; text_color = "#e0f7fa"
    user_bg = "#333"; ai_bg = "#1e88e5"
    user_text = "#f1f8e9"; ai_text = "#ffffff"
else:
    bg_color = "#ffffff"; text_color = "#000000"
    user_bg = "#fff3e0"; ai_bg = "#e3f2fd"
    user_text = "#4e342e"; ai_text = "#0d47a1"

# Apply styling
st.markdown(f"""
<style>
html, body, [class*="css"]  {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
}}

.stChatMessage {{
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 12px;
    font-size: 16px;
    line-height: 1.6;
    animation: fadeIn 0.5s ease-in-out;
    transition: background 0.3s ease;
}}

div.stChatMessage:nth-child(even) {{
    background-color: {ai_bg};
    color: {ai_text};
}}

div.stChatMessage:nth-child(odd) {{
    background-color: {user_bg};
    color: {user_text};
}}

div[data-testid="stChatMessage-avatar"] {{
    margin-right: 10px;
}}

button {{
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 16px;
    transition: all 0.3s ease;
}}

button:hover {{
    background-color: #aed581 !important;
    color: black !important;
}}

h1, h2, h3 {{
    color: {"#80deea" if theme_mode else "#135589"};
}}

@keyframes fadeIn {{
  0% {{ opacity: 0; transform: translateY(10px); }}
  100% {{ opacity: 1; transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)
# ========== SIDEBAR NAV ==========
with st.sidebar:
    st.sidebar.image("doctor photo.png", use_container_width=True)
    selected = option_menu("Menu Options",
                           ["Home", "Medical Report Summarization", "Dr. HeAlify Bot"],
                           icons=["house", "file-earmark-medical-fill", "robot"],
                           default_index=0)

# ========== FILE SERVER ==========
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def run_flask():
    app = Flask(__name__)
    @app.route("/uploads/<filename>")
    def uploaded_file(filename):
        return send_from_directory(UPLOAD_FOLDER, filename)
    app.run(port=8000)

if "flask_thread" not in st.session_state:
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    st.session_state['flask_thread'] = flask_thread

# ========== VOICE OUTPUT ==========
def speak_text_with_controls(text):
    js_code = f"""
    <script>
        if (!window.speechState) {{
            window.speechState = {{
                msg: null, synth: window.speechSynthesis, voices: [],
                text: {repr(text)}, voiceIndex: 0, rate: 1, pitch: 1
            }};
        }} else {{
            window.speechState.text = {repr(text)};
        }}
        function updateSpeechMessage() {{
            const msg = new SpeechSynthesisUtterance(window.speechState.text);
            msg.voice = window.speechState.voices[window.speechState.voiceIndex];
            msg.rate = window.speechState.rate;
            msg.pitch = window.speechState.pitch;
            window.speechState.msg = msg;
        }}
        function populateVoices() {{
            window.speechState.voices = window.speechSynthesis.getVoices();
            const dropdown = document.getElementById('voiceSelect');
            dropdown.innerHTML = "";
            window.speechState.voices.forEach((voice, i) => {{
                const option = document.createElement('option');
                option.value = i;
                option.text = voice.name + " (" + voice.lang + ")";
                dropdown.appendChild(option);
            }});
        }}
        function playSpeech() {{
            if (!window.speechState.msg || window.speechSynthesis.speaking || window.speechSynthesis.paused) return;
            speechSynthesis.speak(window.speechState.msg);
        }}
        function pauseSpeech() {{
            if (speechSynthesis.speaking && !speechSynthesis.paused) speechSynthesis.pause();
        }}
        function resumeSpeech() {{
            if (speechSynthesis.paused) speechSynthesis.resume();
        }}
        function stopSpeech() {{
            if (speechSynthesis.speaking || speechSynthesis.paused) {{
                speechSynthesis.cancel();
                updateSpeechMessage();
            }}
        }}
        window.addEventListener('DOMContentLoaded', () => {{
            if (!document.getElementById('speech-controls')) {{
                let container = document.createElement('div');
                container.id = 'speech-controls';
                container.innerHTML = `
                    <label>🎤 Voice:</label>
                    <select id='voiceSelect' onchange='window.speechState.voiceIndex = this.value; updateSpeechMessage();'></select><br>
                    <label>⚙️ Rate: <span id='rateVal'>1</span></label>
                    <input type='range' min='0.5' max='2' step='0.1' value='1' id='rateSlider'
                        oninput='window.speechState.rate = parseFloat(this.value);
                                 document.getElementById("rateVal").innerText = this.value;
                                 updateSpeechMessage();'><br>
                    <label>🎶 Pitch: <span id='pitchVal'>1</span></label>
                    <input type='range' min='0' max='2' step='0.1' value='1' id='pitchSlider'
                        oninput='window.speechState.pitch = parseFloat(this.value);
                                 document.getElementById("pitchVal").innerText = this.value;
                                 updateSpeechMessage();'><br>
                    <button onclick='playSpeech()'>▶️ Play</button>
                    <button onclick='pauseSpeech()'>⏸ Pause</button>
                    <button onclick='resumeSpeech()'>▶️ Resume</button>
                    <button onclick='stopSpeech()'>⏹ Stop</button>
                `;
                document.body.appendChild(container);
                populateVoices();
                speechSynthesis.onvoiceschanged = populateVoices;
                updateSpeechMessage();
            }}
        }});
    </script>
    """
    components.html(js_code, height=320)

# ========== PDF GENERATOR ==========
def generate_pdf(chat_text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text_object = pdf.beginText(40, height - 40)
    text_object.setFont("Helvetica", 12)
    for line in chat_text.split("\n"):
        text_object.textLine(line)
        if text_object.getY() < 40:
            pdf.drawText(text_object)
            pdf.showPage()
            text_object = pdf.beginText(40, height - 40)
            text_object.setFont("Helvetica", 12)
    pdf.drawText(text_object)
    pdf.save()
    buffer.seek(0)
    return buffer
# ===========================
# HOME PAGE
# ===========================
if selected == 'Home':
    st.title("🏠 Welcome to Dr. HeAlify Doc-Agent")
    st.image("HeAlify main logo.png", use_container_width=True)
    st.markdown("""
        Dr. HeAlify is your personal AI-powered healthcare assistant.  
        🤖 Ask medical questions, 🩺 check symptoms, 📄 summarize reports, and more.
    """)

# ===========================
# MEDICAL REPORT SUMMARIZATION
# ===========================
elif selected == 'Medical Report Summarization':
    st.title("📄 Medical Report Summarization")
    st.markdown("Upload your medical report (PDF) and receive a professional summary.")

    # Load Groq LLM for summarization
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    prompt_template = """
    You are a medical assistant AI that helps summarize patient medical reports.
    <context>
    {context}
    </context>
    Please generate a helpful and readable summary for a general user.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    uploaded_file = st.file_uploader("📎 Upload your medical report PDF", type="pdf")

    # Initialize stateful objects once
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if uploaded_file:
        filename = secure_filename(uploaded_file.name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            st.error("❌ Failed to read the PDF.")
            st.stop()

        # Chunk and embed
        final_documents = st.session_state.text_splitter.split_documents(docs[:20])
        if not final_documents:
            st.error("❌ No valid content found.")
            st.stop()

        vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        context = {'input': prompt_template, 'context': docs}
        response = retrieval_chain.invoke(context)

        st.markdown("### ✅ Summary Result")
        st.write(response['answer'])

        st.download_button("📥 Download Summary", data=response['answer'],
                           file_name="Medical_Report_Summary.txt", mime="text/plain")
# ===========================
# DR. HEALIFY BOT
# ===========================
elif selected == 'Dr. HeAlify Bot':
    st.title("🤖 Dr. HeAlify Bot")
    st.image("chatbot photo.png", use_container_width=True)

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    input_prompt = "You are a world-class virtual doctor named Dr. HeAlify..."

    # Language translation options
    lang_map = {
        "English": "en", "Hindi": "hi", "Spanish": "es",
        "French": "fr", "German": "de", "Tamil": "ta", "Telugu": "te"
    }
    selected_language = st.selectbox("🌐 Select Language", list(lang_map.keys()))
    target_lang = lang_map[selected_language]

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_log_text" not in st.session_state:
        st.session_state["chat_log_text"] = ""

    # Suggested prompts
    st.markdown("### 💡 Quick Suggestions")
    suggestions = [
        "What are the symptoms of diabetes?",
        "How to reduce high blood pressure?",
        "Side effects of paracetamol?",
        "Healthy BMI range?",
        "Improve sleep quality?",
        "COVID-19 symptoms?"
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion):
            st.session_state["chat_history"].append({"role": "user", "content": suggestion})
            user_input = suggestion

    # Voice assistant toggle
    recognizer = sr.Recognizer()
    def recognize_speech():
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                st.info("🎙 Listening...")
                audio = recognizer.listen(source)
                return recognizer.recognize_google(audio)
            except:
                return ""
    use_voice_assistant = st.toggle("🎧 Enable Voice Assistant")
    user_input = recognize_speech() if use_voice_assistant else st.chat_input("Type your question...")

    # Show chat history with avatars
    for msg in st.session_state["chat_history"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    if user_input:
        try:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        except:
            translated_input = user_input

        st.session_state["chat_log_text"] += f"👤 You ({selected_language}): {user_input}\n"
        with st.spinner("🤖 Dr. HeAlify is typing..."):
            gemini_response = model.generate_content([input_prompt, translated_input]).text

        try:
            translated_response = GoogleTranslator(source='en', target=target_lang).translate(gemini_response)
        except:
            translated_response = gemini_response

        st.session_state["chat_history"].append({"role": "assistant", "content": translated_response})
        st.chat_message("assistant", avatar="🤖").write(translated_response)
        st.session_state["chat_log_text"] += f"🤖 Dr. HeAlify: {translated_response}\n\n"
        speak_text_with_controls(translated_response)

    # Chat downloads
    st.download_button("💾 Download Chat (TXT)", data=st.session_state["chat_log_text"],
                       file_name="HeAlify_Chat.txt", mime="text/plain")
    st.download_button("📄 Download Chat (PDF)", data=generate_pdf(st.session_state["chat_log_text"]),
                       file_name="HeAlify_Chat.pdf", mime="application/pdf")

    # Chat summarizer
    if st.button("📝 Summarize This Conversation"):
        summary_prompt = f"Summarize this conversation:\n{st.session_state['chat_log_text']}"
        try:
            summary_response = model.generate_content(summary_prompt).text
            st.markdown("### 📋 Chat Summary")
            st.markdown(summary_response)
            speak_text_with_controls(summary_response)
            st.download_button("📥 Download Summary", data=summary_response,
                               file_name="Dr_HeAlify_Summary.txt", mime="text/plain")
        except Exception:
            st.error("❌ Failed to generate summary.")

    # Symptom checker
    st.markdown("## 🩺 Symptom Checker")
    symptom_input = st.text_input("🔎 Enter your symptoms (comma-separated):")
    if st.button("🔍 Check Conditions"):
        if symptom_input.strip():
            with st.status("📝 Gathering symptoms..."):
                st.write("🧠 Analyzing...")
                st.write("✅ Almost done...")
            symptom_prompt = f"""
            You are a medical assistant. Based on these symptoms: "{symptom_input}",
            suggest 2-3 possible conditions and indicate whether a doctor visit is recommended.
            """
            try:
                result = model.generate_content(symptom_prompt).text
                st.markdown("### 🧾 Possible Conditions")
                st.markdown(result)
                speak_text_with_controls(result)
            except:
                st.error("❌ Could not generate diagnosis.")
        else:
            st.warning("Please enter symptoms to continue.")
