import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Get the path to Google Application Credentials
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path is None:
    credentials_path = "C:\\Users\\asus\\Downloads\\saathi-439108-2866ecb350dc.json"
print("GOOGLE_APPLICATION_CREDENTIALS:", credentials_path)

if not credentials_path:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or is None.")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

genai.configure(api_key=os.getenv("AIzaSyCpzURfYqs9TbCw7yncdMt09dsj0bsvkW0"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def generate_quiz_questions_from_text(text, noq):
    prompt_template = """
    Generate {NOQ} quiz questions based on the following content:\n{context}\n
For each question, provide 4 answer options labeled A, B, C, and D, with one correct answer. Format the output as follows:

Question 1: [Your question here]\n
\n A. [Option A]\n
\n B. [Option B]\n
\n C. [Option C]\n
\n D. [Option D]\n

Question 2: [Your question here]\n
\n A. [Option A]\n
\n B. [Option B]\n
\n C. [Option C]\n
\n D. [Option D]\n

... and so on.

Answer Key:\n
1. [Question 1 number] - [Correct option]\n
2. [Question 2 number] - [Correct option]\n
...


    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "NOQ"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    document = Document(page_content=text, metadata={})
    response = chain({"input_documents": [document], "NOQ": noq})
    return response["output_text"]


def generate_random_questions_from_topic(topic, noq):
    prompt_template = """
    Generate {NOQ} quiz questions based on the following content:\n{context}\n
For each question, provide 4 answer options labeled A, B, C, and D, with one correct answer. Format the output as follows:

Question 1: [Your question here]\n
\n A. [Option A]\n
\n B. [Option B]\n
\n C. [Option C]\n
\n D. [Option D]\n

Question 2: [Your question here]\n
\n A. [Option A]\n
\n B. [Option B]\n
\n C. [Option C]\n
\n D. [Option D]\n

... and so on.

Answer Key:\n
1. [Question 1 number] - [Correct option]\n
2. [Question 2 number] - [Correct option]\n
...


    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "NOQ"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    document = Document(page_content=topic, metadata={})
    response = chain({"input_documents": [document], "NOQ": noq})
    return response["output_text"]


def main():
    st.set_page_config("Chat PDF and Quiz Generator")
    st.header("Generate Quiz Questions from PDF or Topic 🎓")

    option = st.radio("Select Input Type:", ("PDF", "Topic"))
    noq = st.number_input("Enter the Number of Questions to Generate:", min_value=1, max_value=20, value=5)

    if option == "PDF":
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("Generate Quiz Questions"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    questions = generate_quiz_questions_from_text(raw_text, noq)
                    st.success("Quiz Questions Generated!")
                    st.write(questions)
            else:
                st.warning("Please upload a PDF file.")

    elif option == "Topic":
        topic = st.text_input("Enter the Topic:")
        
        if st.button("Generate Questions"):
            if topic:
                with st.spinner("Generating Questions..."):
                    questions = generate_random_questions_from_topic(topic, noq)
                    st.success("Random Questions Generated!")
                    st.write(questions)
            else:
                st.warning("Please enter a topic.")

if __name__ == "__main__":
    main()
