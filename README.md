# InsightX: AI-Powered Document & Visual Assistant

InsightX is an advanced AI-driven solution designed to extract, analyze, and summarize content from PDFs and images. Leveraging state-of-the-art OCR, semantic search, and data visualization techniques, InsightX empowers users to gain actionable insights from their documents quickly and efficiently.

---

## Table of Contents

- [Deployment](#deployment)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)

---

## Deployment

[https://ai-insightx.streamlit.app/](https://ai-insightx.streamlit.app/)

---

## Features

- **Document Processing**  
  - Extract text from PDFs and images using advanced OCR.
  
- **Metadata Extraction**  
  - Retrieve key metadata for enhanced document insights.
  
- **Keyword Analysis**  
  - Identify frequent terms and visualize them with bar charts and word clouds.
  
- **Entity Recognition**  
  - Automatically extract names, dates, organizations, invoice numbers, and totals.
  
- **Document Classification**  
  - Automatically categorize documents (e.g., Invoices, Reports, Contracts).
  
- **AI-Powered Q&A**  
  - Leverage semantic search for context-based querying of document content.
  
- **Semantic Search**  
  - Find and highlight relevant document sections using vector-based search.
  
- **Visual Insights**  
  - Generate dynamic visualizations such as pie charts and bar graphs.
  
- **Summarization**  
  - Produce concise summaries of lengthy documents.
  
- **Text-to-Speech**  
  - Convert text into speech for improved accessibility.

---

## System Requirements

- **Python:** Version 3.7 or later  
- **Libraries:**  
  - Streamlit  
  - Additional dependencies listed in `requirements.txt`

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KaviyaaPriyadharshini/AI-InsightX.git
cd AI-InsightX
```

### 2. Create a Virtual Environment (Optional but Recommended)
For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```
For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration
Before running the application, verify and update the required configuration settings:

Open the .env file and set your API keys. For example:

```bash
GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Document Upload & Processing

- **Upload Files:**  
  Use the provided uploaders to submit your documents. You can upload PDFs, images, or both based on your selected mode.

- **Process Files:**  
  Click the **"
  Process Files"** button to extract text, metadata, and visual data from the uploaded files. The application will combine and process the content for further analysis.

### Interacting with InsightX

- **Ask Questions:**  
  Type your query into the text input field. The AI engine leverages semantic search to provide context-based answers from the processed document content.

- **Quick Actions:**  
  Utilize the quick action buttons to:
  - **Extract Entities:** Retrieve key entities (names, dates, organizations, etc.) from the content.
  - **Generate Summaries:** Create concise summaries of lengthy documents.
  - **Visualize Data:** Produce visual insights like charts and word clouds to represent keyword frequency and other metrics.
  - **Text-to-Speech:** Convert the AI's textual response into speech for accessibility.

- **View Insights:**  
  After processing, review the combined document insights including metadata, extracted text, and visualizations directly on the interface.

---
