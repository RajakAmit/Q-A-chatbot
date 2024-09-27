import os
import re
from datetime import datetime

import google.generativeai as genai
import streamlit as st
from dateutil import parser as date_parser
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from pydantic import BaseModel, Field, ValidationError
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Class for booking appointments
class AppointmentBooking(BaseModel):
    name: str = Field(..., description="Full name of the user")
    phone_number: str = Field(..., description="Phone number in string format")
    email: str = Field(..., description="Email address")
    date: str = Field(..., description="Appointment date in format YYYY-MM-DD")

# Function to extract text from PDF documents
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store for text embedding
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load conversational chain for answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say 'answer is not available in the context'.\n\n
    Context:\n {context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function for answering user queries
def user_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to simulate booking an appointment
def book_appointment_tool(name: str, phone_number: str, email: str, date: str):
    return {
        "message": f"Appointment booked for {name} on {date}. Confirmation sent to {email}."
    }

# Validate phone number
def validate_phone_number(phone_number):
    return re.fullmatch(r'\d{10}', phone_number)

# Validate email address
def validate_email(email):
    return re.fullmatch(r'[^@]+@[^@]+\.[^@]+', email)

# Convert natural language date input to proper date format (YYYY-MM-DD)
def convert_to_date(preferred_date):
    try:
        parsed_date = date_parser.parse(preferred_date, fuzzy=True)
        return parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        return None

# Main Streamlit app
def main():
    st.set_page_config("Chat PDF and Book Appointments")
    st.header("Chat with PDF using Gemini💁 and Book Appointments")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    create_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")

    # User input for question
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question.lower() == "call me":
        st.subheader("Please provide your details to book an appointment:")

        # Appointment form
        with st.form(key="appointment_form"):
            name = st.text_input("Name", placeholder="Enter your full name")
            phone_number = st.text_input("Phone Number", placeholder="Enter your phone number")
            email = st.text_input("Email", placeholder="Enter your email")
            preferred_date = st.text_input("Preferred Date (e.g. 'next Monday')", placeholder="Enter your preferred date")

            # Submit button for form
            submit_button = st.form_submit_button("Book Appointment")

        if submit_button:
            if not validate_phone_number(phone_number):
                st.error("Invalid phone number. Please enter a 10-digit phone number.")
            elif not validate_email(email):
                st.error("Invalid email address. Please enter a valid email.")
            else:
                converted_date = convert_to_date(preferred_date)
                if converted_date:
                    try:
                        appointment = AppointmentBooking(
                            name=name,
                            phone_number=phone_number,
                            email=email,
                            date=converted_date
                        )
                        confirmation = book_appointment_tool(
                            appointment.name,
                            appointment.phone_number,
                            appointment.email,
                            appointment.date
                        )
                        st.success(confirmation["message"])
                    except ValidationError as e:
                        st.error(f"Validation error: {e}")
                else:
                    st.error("Invalid date format. Please enter a valid date (e.g., 'next Monday').")
    
    # Answering user questions based on the PDF content
    if user_question and user_question.lower() != "call me":
        if pdf_docs:
            with st.spinner("Processing your question..."):
                answer = user_query(user_question)
                st.write(answer)
        else:
            st.error("Please upload a PDF document first to ask questions.")

if __name__ == "__main__":
    main()






