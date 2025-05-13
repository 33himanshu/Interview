from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from typing import List, Dict
import PyPDF2
import io
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="CV Interview Assistant")

# Load API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Store conversation sessions
conversation_sessions = {}

class QuestionResponse(BaseModel):
    questions: List[str]
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(
    cv_file: UploadFile = File(...),
    job_description: str = Form(...),
    num_questions: int = Form(5)
):
    # Extract text from CV
    cv_text = await extract_text_from_cv(cv_file)
    
    # Generate questions using Gemini
    questions, session_id = await generate_interview_questions(cv_text, job_description, num_questions)
    
    return QuestionResponse(questions=questions, session_id=session_id)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Generate questions first.")
    
    # Get the conversation chain for this session
    conversation = conversation_sessions[session_id]
    
    # Continue the conversation
    response = conversation.invoke({"input": user_message})
    
    return ChatResponse(response=response["response"])

async def extract_text_from_cv(cv_file: UploadFile) -> str:
    content = await cv_file.read()
    
    if cv_file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif cv_file.filename.endswith(('.docx', '.doc')):
        # Add docx handling if needed
        raise HTTPException(status_code=400, detail="DOCX format not supported yet")
    else:
        return content.decode('utf-8')

async def generate_interview_questions(cv_text: str, job_description: str, num_questions: int) -> tuple[List[str], str]:
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GOOGLE_API_KEY)
    
    # Create prompt template for question generation
    prompt = ChatPromptTemplate.from_template(
        """You are an expert interviewer. Based on the CV and job description provided, 
        generate {num_questions} relevant technical interview questions.
        
        CV: {cv_text}
        
        Job Description: {job_description}
        
        Generate questions that assess the candidate's skills, experience, and fit for the role.
        Return only the questions as a numbered list without any additional text."""
    )
    
    # Generate questions
    chain = prompt | llm
    response = chain.invoke({
        "cv_text": cv_text,
        "job_description": job_description,
        "num_questions": num_questions
    })
    
    # Parse the response to extract questions
    questions_text = response.content
    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    questions = questions[:num_questions]  # Ensure we have the requested number of questions
    
    # Create a conversation memory with context
    memory = ConversationBufferMemory()
    
    # Add context to the memory
    memory.chat_memory.add_user_message(
        f"I'm an interviewer with the following CV: {cv_text[:500]}... "
        f"and job description: {job_description[:500]}... "
        f"I've prepared these questions: {', '.join(questions)}"
    )
    memory.chat_memory.add_ai_message(
        "I'll help you conduct this interview. You can ask me to elaborate on any question "
        "or provide additional questions based on the candidate's responses."
    )
    
    # Create a conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Generate a session ID
    session_id = os.urandom(16).hex()
    
    # Store the conversation
    conversation_sessions[session_id] = conversation
    
    return questions, session_id