\# InsightX: AI-Powered Document & Visual Assistant

InsightX is an advanced AI-driven solution designed to extract, analyze,
and summarize content from PDFs and images. Leveraging state-of-the-art
OCR, semantic search, and data visualization techniques, InsightX
empowers users to gain actionable insights from their documents quickly
and efficiently.

\-\--

\## Table of Contents

\- \[Overview\](#overview) - \[Features\](#features) - \[System
Requirements\](#system-requirements) - \[Installation\](#installation) -
\[Configuration\](#configuration) - \[Usage\](#usage) -
\[Deployment\](#deployment) - \[Contribution
Guidelines\](#contribution-guidelines) - \[License\](#license) -
\[Contact\](#contact)

\-\--

\## Overview

InsightX streamlines document processing by extracting text from PDFs
and images, performing entity recognition and semantic search, and
providing interactive visual insights. It is ideal for businesses and
individuals looking to automate data extraction and gain meaningful
insights from unstructured data.

\-\--

\## Features

\- \*\*Document Processing\*\*  - Extract text from PDFs and images
using advanced OCR.

\- \*\*Metadata Extraction\*\*  - Retrieve key metadata for enhanced
document insights.

\- \*\*Keyword Analysis\*\*  - Identify frequent terms and visualize
them with bar charts and word clouds.

\- \*\*Entity Recognition\*\*  - Automatically extract names, dates,
organizations, invoice numbers, and totals.

\- \*\*Document Classification\*\*  - Automatically categorize documents
(e.g., Invoices, Reports, Contracts).

\- \*\*AI-Powered Q&A\*\*  - Leverage semantic search for context-based
querying of document content.

\- \*\*Semantic Search\*\*  - Find and highlight relevant document
sections using vector-based search.

\- \*\*Visual Insights\*\*  - Generate dynamic visualizations such as
pie charts and bar graphs.

\- \*\*Summarization\*\*  - Produce concise summaries of lengthy
documents.

\- \*\*Text-to-Speech\*\*  - Convert text into speech for improved
accessibility.

\-\--

\## System Requirements

\- \*\*Python:\*\* Version 3.7 or later  - \*\*Libraries:\*\*  -
Streamlit  - gTTS  - Pillow  - Additional dependencies listed in
\`requirements.txt\`

\-\--

\## Installation

\### 1. Clone the Repository

\`\`\`bash git clone
https://github.com/KaviyaaPriyadharshini/AI-InsightX.git cd AI-InsightX
2. Create a Virtual Environment (Optional but Recommended) For
macOS/Linux:

bash Copy Edit python -m venv venv source venv/bin/activate For Windows:

bash Copy Edit python -m venv venv venv\\Scripts\\activate 3. Install
Dependencies bash Copy Edit pip install -r requirements.txt
Configuration Before running the application, verify and update the
required configuration settings:

API Keys: Open the .env file and set your API keys. For example:

dotenv Copy Edit GOOGLE_API_KEY=your_google_api_key Additional
Environment Variables: Add or update any environment variables as needed
for your deployment environment.

Usage Document Upload & Processing Upload Files:

PDFs: Use the PDF uploader to submit one or multiple files.

Images: Use the image uploader to submit visual files (png, jpg, jpeg).

Process Files: Click on \"✨ Process Files\" to extract text and
metadata from the uploaded documents.

Interacting with InsightX Ask Questions: Type your query into the text
input. The AI engine will provide context-based responses by searching
through the processed content.

Quick Actions: Utilize the quick action buttons to extract entities,
generate summaries, create visual insights, and more.

Text-to-Speech: Convert AI responses to speech by clicking the \"🎙️ Get
Voice Assistant\" button.

Deployment Local Deployment Run the Streamlit application locally with
the following command:

bash Copy Edit streamlit run app.py Deploying on Streamlit Cloud Push to
GitHub: Ensure your repository is up-to-date on GitHub.

Set Up on Streamlit Cloud:

Log in to Streamlit Cloud.

Click \"New App\", select your repository and branch, and set app.py as
the main file.

Deploy your app.

Environment Variables: Configure any necessary environment variables
directly in the Streamlit Cloud dashboard.

Contribution Guidelines We welcome contributions to enhance InsightX. To
contribute:

Fork the repository.

Create a feature branch (git checkout -b feature/your-feature-name).

Commit your changes and push your branch.

Open a pull request with a detailed description of your changes.

Follow our coding standards and document any new features.

License This project is licensed under the MIT License.

Contact For inquiries, support, or further information, please contact:

Project Maintainer: Kaviyaa Priyadharshini

Email: your-email@example.com
