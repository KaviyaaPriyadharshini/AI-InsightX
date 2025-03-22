
# InsightX: AI-Powered Document & Visual Assistant

InsightX is an AI-powered tool for extracting, analyzing, and summarizing text from PDFs and images, featuring entity recognition, semantic search, and interactive insights.
## Features


- Document Processing:  Extract text from PDFs and images using OCR.
- Metadata Extraction: Retrieve document metadata for insights.
- Keyword Analysis: Identify frequent words and visualize them in bar charts and word clouds.
- Entity Recognition: Extract names, dates, organizations, invoice numbers, and totals.
- Document Classification: Automatically categorize documents (e.g., Invoice, Report, Contract).
- AI-Powered Q&A: Ask questions and get answers based on extracted document content.
- Semantic Search: Find relevant document sections using vector-based search.
- Visual Insights: Generate pie charts and bar graphs for data representation.
- Summarization: Generate concise summaries of long documents.
- Text-to-Speech: Convert extracted text into speech for accessibility.
## Usage

 - Upload Documents
 - View Insights
 - Visualize Data
 - Ask Questions
 - Perform Semantic Search
 - Summarize Content
 - Text-to-Speech
   
## Installation

1. Clone the Repository:

```bash
git clone https://github.com/KaviyaaPriyadharshini/AI-InsightX
cd AI-InsightX
```
2. Create a Virtual Environment (Optional but Recommended):

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
3. Install Dependencies:

```bash
pip install -r requirements.txt
```
4. Set Up API Keys:
Open .env and verify that the required API keys are correctly set.
```bash
GOOGLE_API_KEY=your_google_api_key
```
4. Run the Streamlit App:

```bash
streamlit run app.py
```
              
