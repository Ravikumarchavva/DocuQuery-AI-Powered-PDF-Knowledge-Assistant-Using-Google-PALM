# DocuQuery-AI-Powered-PDF-Knowledge-Assistant-Using-Google-Gemini-API

This project allows you to upload a PDF document, extract its text, and interact with it through a question-answering system powered by Google's Gemini API. It utilizes **LangChain**, **Streamlit**, **FAISS**, and **Google Generative AI Embeddings** to process the text and provide detailed answers to user queries.

## Features

- Upload and process multiple PDF files.
- Extract and split text into chunks.
- Store text embeddings in FAISS for efficient similarity search.
- Use Google Gemini for detailed question-answering based on the PDF content.
- Interactive user interface using Streamlit.

## Getting Started

To run the project locally, follow these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Google API Key (for accessing Gemini API)
- Required Python packages

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Ravikumarchavva/DocuQuery-AI-Powered-PDF-Knowledge-Assistant-Using-Google-PALM.git
    cd DocuQuery-AI-Powered-PDF-Knowledge-Assistant-Using-Google-PALM.git
    ```

2. **Create a virtual environment (optional but recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:

    Create a `.env` file in the root directory of the project and add your Google API key:

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key_here
    ```

    Replace `your_google_api_key_here` with your actual Google API key.

### Running the Application

Once everything is set up, run the following command to start the Streamlit app:

```bash
streamlit run app.py
