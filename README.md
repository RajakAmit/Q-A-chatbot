# Q-A-chatbot


# PDF Conversational Interface with Appointment Booking

This project allows users to interact with PDF documents using a conversational interface powered by Google's Gemini AI (Generative AI). It also includes functionality for booking appointments through a form, with natural language date parsing and input validation.

## 🔧 Key Features:
- **Upload PDFs**: Users can upload one or more PDF documents, and the app extracts text for further interaction.
- **Conversational Q&A**: Users can ask questions based on the content of the uploaded PDFs, and the system will retrieve relevant answers using FAISS-based vector search and Google Generative AI (Gemini) for question-answering.
- **Appointment Booking**: Users can book appointments by providing details like name, phone number, email, and preferred date. The app validates the inputs and confirms the booking.

## 📚 Libraries & Technologies:
- **Streamlit**: For creating the interactive web interface.
- **Google Generative AI (Gemini)**: Powering the question-answering model and embedding generation.
- **FAISS**: For creating vector stores to retrieve answers from PDF content.
- **PyPDF2**: For extracting text from PDF documents.
- **LangChain**: For handling the question-answering chain.
- **Pydantic**: For validating user input (appointment booking).
- **dotenv**: For managing environment variables securely.

## 🚀 How It Works:
1. **Upload PDF Documents**: Users upload PDFs via the sidebar, and the app extracts text using PyPDF2.
2. **Text Processing**: The extracted text is split into chunks for better retrieval performance.
3. **Conversational Interface**: Users can ask questions related to the PDF content, and the app uses embeddings and a conversational chain to return accurate answers.
4. **Appointment Booking**: The app allows users to book appointments by parsing natural language dates (e.g., "next Monday") and validating inputs like phone numbers and emails.

## 📂 Project Structure:
- `app.py`: Main application file containing the Streamlit interface and core logic.
- `.env`: Environment variables file containing API keys for Google Generative AI.
- `requirements.txt`: Dependencies required to run the project.
- `faiss_index`: Local vector store for efficient retrieval of text chunks.


