import streamlit as st
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

# Setup tesseract path (adjust this based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Dynamically fetch available models
models = genai.list_models()
available_models = [m.name for m in models if "generateContent" in m.supported_generation_methods]
vision_model_available = "models/gemini-1.5-pro-vision" in available_models
text_model = "models/gemini-1.5-pro" if "models/gemini-1.5-pro" in available_models else available_models[0]

# ---------------------- PDF Handling ----------------------
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

# ---------------------- Image Handling ----------------------
def extract_text_from_images(uploaded_images):
    extracted_text = ""
    for image_file in uploaded_images:
        image = Image.open(image_file)
        
        # OCR extraction
        ocr_text = pytesseract.image_to_string(image)
        extracted_text += f"\n[OCR Output]\n{ocr_text}\n"

        if vision_model_available:
            # Optional: If Vision model available, send image to Gemini Vision
            vision_model = genai.GenerativeModel("models/gemini-1.5-pro-vision")
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{image_base64}"
            response = vision_model.generate_content([image_data_uri, "Extract key information, summarize."])
            extracted_text += f"\n[Gemini Vision Insights]\n{response.text}\n"
        else:
            extracted_text += f"\n[Fallback: Text model will handle OCR text]\n"
    return extracted_text

# ---------------------- Chunk + Vector Store ----------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ---------------------- QA Chain ----------------------
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

# ---------------------- Chat History ----------------------
def display_chat_history():
    st.subheader("📜 Chat History")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(
                f"""
                <div style='padding: 10px; margin: 10px 0; background: #f4f4f4; border-radius: 10px;'>
                    <strong>Q: {chat['question']}</strong>
                    <br>
                    <span style='color: #333;'>💬 {chat['answer']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No chat history available. Start asking questions!")

def download_chat_history():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        history_md = ""
        for chat in st.session_state.chat_history:
            history_md += f"### Q: {chat['question']}\n**A:** {chat['answer']}\n\n"
        b64 = base64.b64encode(history_md.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.md">📥 Download Chat History</a>'
        st.markdown(href, unsafe_allow_html=True)

# ---------------------- MAIN APP ----------------------
def main():
    st.set_page_config("Multi PDF + Image Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF + Image 🖼️ 📚 - Chat Agent 🤖")

    mode = st.sidebar.radio("Select Mode", ["PDF + Images", "PDF only", "Images only"])

    with st.sidebar:
        st.write("---")
        st.title("📁 Upload Section")
        pdf_docs = None
        uploaded_images = None

        if mode in ["PDF + Images", "PDF only"]:
            pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        if mode in ["PDF + Images", "Images only"]:
            uploaded_images = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                combined_text = ""
                if pdf_docs:
                    st.info("Extracting from PDFs...")
                    combined_text += get_pdf_text(pdf_docs)
                if uploaded_images:
                    st.info("Extracting from Images...")
                    combined_text += extract_text_from_images(uploaded_images)
                text_chunks = get_text_chunks(combined_text)
                get_vector_store(text_chunks)
                st.success("Processing Completed ✅")

        if st.button("🔄 Reset Session"):
            st.session_state.chat_history = []
            st.rerun()
        st.write("---")

    user_question = st.text_input("Ask your Question 📝")
    if user_question:
        answer = user_input(user_question)
        st.write("Reply: ", answer)

    display_chat_history()
    download_chat_history()

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #B2BEB5; padding: 15px; text-align: center;">
            PDF & IMAGE CHATBOT ❤️ [ DIV - KAV - HAR - IND ]
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()  