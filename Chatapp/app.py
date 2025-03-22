from utils import display_chat_history,get_conversational_chain,get_keyword_frequencies,get_pdf_text,get_text_chunks,get_vector_store,show_insights_dashboard,semantic_search,summarize_text,classify_document_type,text_to_speech,user_input,user_input_with_context,extract_entities,extract_keywords,extract_text_easyocr,highlight_keywords
from chat_history import save_chat_history,load_chat_history
import streamlit as st
from PIL import Image

def main():
    st.set_page_config("InsightX", page_icon="🔍", layout="wide")

    custom_css = """
    <style>
        .sidebar .sidebar-content { background-color: #262730; color: white; }
        .stButton>button { width: 100%; }
        .stMarkdown a { color: #6C63FF !important; }
        .block-container { padding-top: 2rem; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("📚 InsightX: AI-Powered Document & Visual Assistant")

    with st.sidebar:
        mode = st.sidebar.radio("🚀 **Choose Mode**", ["PDF + Images", "PDF only", "Images only"])
        st.subheader("📂 Upload Your Files")

        pdf_docs = None
        uploaded_images = None

        if mode in ["PDF + Images", "PDF only"]:
            pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if mode in ["PDF + Images", "Images only"]:
            uploaded_images = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

        if "combined_text" not in st.session_state:
            st.session_state["combined_text"] = None
        if "matched_chunks" not in st.session_state:
            st.session_state["matched_chunks"] = None
        if "show_matched_sections" not in st.session_state:
            st.session_state["show_matched_sections"] = False

        if st.button("✨ Process Files"):
            if not pdf_docs and not uploaded_images:
                st.warning("⚠️ Please upload at least one PDF or Image before processing.")
            else:
                with st.spinner("🔄 Processing your documents..."):
                    combined_text_pdf = ""
                    combined_text_image=""
                    if pdf_docs:
                        st.info("🔍 Extracting from PDFs...")
                        combined_text_pdf += get_pdf_text(pdf_docs)

                    if uploaded_images:
                        st.info("🔍 Extracting from Images...")
                        extracted_texts = []
                        for uploaded_file in uploaded_images:
                            image = Image.open(uploaded_file)  
                            text = extract_text_easyocr(image)
                            extracted_texts.append(text)
                        combined_text_image = "\n".join(extracted_texts)
                    combined_text=combined_text_pdf+combined_text_image
                    if not combined_text.strip():
                        st.error("⚠️ No text could be extracted! Please upload valid files.")
                    else:
                        st.info("🧠 Auto-Categorizing...")

                        if combined_text_image and combined_text_pdf:
                            doc_type = classify_document_type(combined_text_pdf)
                            st.success(f"🗂 Pdf Document Type: **{doc_type}**")
                            doc_type = classify_document_type(combined_text_image)
                            st.success(f"🗂 Image Type: **{doc_type}**")
                        elif combined_text_image:
                            doc_type = classify_document_type(combined_text_image)
                            st.success(f"🗂 Image Type: **{doc_type}**")
                        elif combined_text_pdf:
                            doc_type = classify_document_type(combined_text_pdf)
                            st.success(f"🗂 Pdf Document Type: **{doc_type}**")

                        st.session_state["combined_text"] = combined_text
                        text_chunks = get_text_chunks(combined_text)
                        get_vector_store(text_chunks)

                        st.success("✅ Processing Complete")

        if st.button("♻️ Reset"):
            st.session_state.clear()
            st.rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input("💬 Ask your question (Semantic Search Active)")

        if user_question and (pdf_docs or uploaded_images):
            selected_source = pdf_docs[0].name if pdf_docs else uploaded_images[0].name

            if not st.session_state.get("combined_text"):
                st.warning("⚠️ Please process documents before asking questions.")
            else:
                answer, matched_chunks = user_input_with_context(user_question)
                st.write("🤖 **AI Response:**", answer)

                if st.button("🎙️ Get Voice Assistant"):
                    text_to_speech(answer)

                st.session_state["matched_chunks"] = matched_chunks
                save_chat_history(selected_source, user_question, answer)

                display_chat_history()

    with col2:
        st.subheader("🎯 Quick Actions")

        if st.session_state["combined_text"]:
                
            with st.container():
                if st.button("🔍 Extract Entities"):
                    entities = extract_entities(st.session_state["combined_text"])
                    st.session_state["entities"] = entities
                    st.json(entities)

            with st.container():
                if st.button("📝 Get Document Summary"):
                    summary = summarize_text(st.session_state["combined_text"])
                    st.info(summary)

            with st.container():
                if st.button("📊 Generate Insights Dashboard"):
                    entities = st.session_state.get("entities")
                    if not entities:
                        st.warning("⚠️ Please extract entities first.")
                    else:
                        show_insights_dashboard(entities, st.session_state["combined_text"])

            with st.container():
                if st.button("📖 Show Matched Sections"):
                    if not st.session_state["matched_chunks"]:
                        st.warning("⚠️ No matched sections found! Ask a question first.")
                    else:
                        st.session_state["show_matched_sections"] = True  

                if st.session_state["show_matched_sections"]:
                    keywords = extract_keywords(user_question)
                    st.markdown("### 🧩 **Matched Document Sections (with Highlights):**")
                    for idx, chunk in enumerate(st.session_state["matched_chunks"]):
                        highlighted = highlight_keywords(chunk, keywords)
                        st.markdown(f"""
                        <div style='padding:10px; margin-bottom:10px; background:#f9f9f9; border-left:4px solid #6C63FF; border-radius:5px;'>
                            <b>Match {idx + 1}:</b><br>
                            <div style="line-height:1.6;">{highlighted}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Upload documents to enable actions.")

    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: white;
            padding: 10px 0;
            border-top: 1px solid #6C63FF;
            text-align: center;
        }
    </style>
    <div class="footer">
        🚀 Built with ❤️ by DIV - KAV - HAR - IND | Powered by Gemini AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
