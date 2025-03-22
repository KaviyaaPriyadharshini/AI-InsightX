import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import os
import io
import base64
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from google.cloud import vision
from gtts import gTTS
from chat_history import load_chat_history
import easyocr

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = genai.list_models()
available_models = [m.name for m in models if "generateContent" in m.supported_generation_methods]
vision_model_available = "models/gemini-1.5-pro-vision" in available_models
text_model = "models/gemini-1.5-pro" if "models/gemini-1.5-pro" in available_models else available_models[0]

def get_pdf_text(pdf_docs):
    text = ""
    metadata_info = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        metadata = pdf_reader.metadata
        metadata_info += f"\n[Metadata] {metadata}\n"
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return metadata_info + text
import numpy as np

def extract_text_easyocr(image):
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    image = image.convert("RGB")
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)  
    return "\n".join(result)

def get_keyword_frequencies(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    frequency = {}
    for word in filtered_words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency

def show_insights_dashboard(entities, combined_text):
    st.subheader("📊 Auto-Generated Insights Dashboard")

    entity_counts = {
        "Dates": len(entities.get("dates", [])),
        "Organizations": len(entities.get("organizations", [])),
        "Names": len(entities.get("names", [])),
        "Invoice Numbers": len(entities.get("invoice_numbers", [])),
        "Totals": len(entities.get("totals", [])),
    }
    entity_counts = {k: v for k, v in entity_counts.items() if v > 0}  
    
    if entity_counts:
        st.markdown("#### 🥧 Entity Distribution (Pie Chart)")
        fig_pie = px.pie(
            names=list(entity_counts.keys()),
            values=list(entity_counts.values()),
            title="Extracted Entities Distribution"
        )
        st.plotly_chart(fig_pie)
    else:
        st.info("No entities found to display pie chart.")

    if entity_counts:
        st.markdown("#### 📊 Entity Count (Bar Graph)")
        fig_bar = px.bar(
            x=list(entity_counts.keys()),
            y=list(entity_counts.values()),
            labels={'x': 'Entity', 'y': 'Count'},
            title="Extracted Entities Count"
        )
        st.plotly_chart(fig_bar)

    st.markdown("#### 🗝 Keyword Frequency (Bar Graph)")
    freq = get_keyword_frequencies(combined_text)
    if freq:
        top_freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True)[:10])
        fig_keywords = px.bar(
            x=list(top_freq.keys()),
            y=list(top_freq.values()),
            labels={'x': 'Keyword', 'y': 'Frequency'},
            title="Top 10 Keywords"
        )
        st.plotly_chart(fig_keywords)
    else:
        st.info("No keywords found for frequency graph.")

    st.markdown("#### ☁️ Word Cloud of Keywords")
    if freq:
        wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate_from_frequencies(freq)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("No words available for word cloud.")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context," don't make it up.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=text_model, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    time.sleep(1)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    answer = response["output_text"]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"question": user_question, "answer": answer})
    return answer

def classify_document_type(text):
    model = genai.GenerativeModel(model_name=text_model)
    prompt = f"Classify this document into one of the types: Invoice, Report, Contract, Manual, Legal Document, Medical Record.\n\nContent:\n{text[:3000]}\n\nType:"
    response = model.generate_content(prompt)
    return response.text.strip()

def extract_entities(text):
    model = genai.GenerativeModel(model_name=text_model)
    prompt = f"Extract key entities (names, dates, invoice numbers, totals, organizations, etc.) from the text below and format as JSON:\n\n{text[:3000]}"
    response = model.generate_content(prompt)
    cleaned_response = re.sub(r"```json|```", "", response.text).strip()
    try:
        parsed_json = json.loads(cleaned_response)
        return parsed_json
    except json.JSONDecodeError:
        return {"raw_output": cleaned_response}

def summarize_text(text):
    model = genai.GenerativeModel(model_name=text_model)
    prompt = f"Summarize the following document briefly:\n\n{text[:3000]}"
    response = model.generate_content(prompt)
    return response.text.strip()

def display_chat_history():
    st.markdown("<h4 style='margin-bottom:10px;'>📜 Chat History</h4>", unsafe_allow_html=True)

    chat_history = load_chat_history()

    if chat_history:
        pdf_grouped = {}
        for pdf, question, answer in chat_history:
            if pdf not in pdf_grouped:
                pdf_grouped[pdf] = []
            pdf_grouped[pdf].append((question, answer))

        for pdf, chats in pdf_grouped.items():
            st.markdown(f"### 📄 Chat from **{pdf}**")
            for question, answer in chats:
                st.markdown(f"""
                <div style='padding: 10px; margin-bottom: 10px; background: #F0F0F5; border-left: 5px solid #6C63FF; border-radius: 8px;'>
                    <b>Q:</b> {question}<br>
                    <b>A:</b> {answer}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No chat history available yet.")

def extract_keywords(query):
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    return keywords

def highlight_keywords(text, keywords):
    highlighted_text = text
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted_text = pattern.sub(
            lambda m: f"<mark style='background: #FFD54F; padding: 2px; border-radius: 3px;'>{m.group(0)}</mark>", 
            highlighted_text
        )
    return highlighted_text

def semantic_search(query, k=3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(query, k=k)
    return docs

def user_input_with_context(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    answer = response["output_text"]
    matched_chunks = [doc.page_content for doc in docs]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"question": user_question, "answer": answer})
    return answer, matched_chunks

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    audio_file = "response.mp3"
    tts.save(audio_file)

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio id="tts-audio" controls>
        <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)