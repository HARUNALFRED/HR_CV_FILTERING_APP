import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from io import StringIO
import os
from PyPDF2 import PdfReader
import docx

# Setup Streamlit page
st.set_page_config(page_title="HR CV Filtering System", layout="wide")

# Title of the application with an icon and description
st.title("HR CV Filtering System")
st.markdown("""
    <style>
    .title {
        color: #4CAF50;
        font-size: 35px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Section for job description
st.header("1. Enter Job Description")
job_description = st.text_area("Paste the job description here:", height=150)

# File upload for multiple CVs
st.header("2. Upload CVs")
uploaded_files = st.file_uploader("Upload CVs", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Function to extract text from PDFs, DOCX, or TXT files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        return uploaded_file.read().decode("utf-8")

# Load pre-trained DistilBERT model and tokenizer from Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to convert text into embeddings using DistilBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Store CVs text in a list
cv_texts = []
cv_names = []
for uploaded_file in uploaded_files:
    file_text = extract_text_from_file(uploaded_file)
    cv_texts.append(file_text)
    cv_names.append(uploaded_file.name)

# Display names of uploaded CVs
if uploaded_files:
    st.subheader("Uploaded CVs:")
    for i, cv_name in enumerate(cv_names):
        st.write(f"{i + 1}. {cv_name}")

# Function to rank CVs based on cosine similarity and return percentage match
def rank_cvs_with_similarity(job_description, cv_texts):
    job_embedding = get_embedding(job_description)
    cv_embeddings = [get_embedding(cv) for cv in cv_texts]
    
    similarities = [cosine_similarity(job_embedding.reshape(1, -1), cv_embedding.reshape(1, -1))[0][0] for cv_embedding in cv_embeddings]
    
    # Normalize similarity to percentage
    percentage_match = [similarity * 100 for similarity in similarities]
    
    ranked_cvs = sorted(zip(percentage_match, similarities, cv_names, cv_texts), reverse=True, key=lambda x: x[0])
    return ranked_cvs

# Sidebar for Filter Selection
st.sidebar.header("Filter Options")
top_n = st.sidebar.slider("Select Top N Candidates", 1, 20, 10)

# Button to trigger ranking and display
if job_description and uploaded_files:
    if st.sidebar.button('Filter and Rank Candidates'):
        st.header("3. Top Qualifying Candidates")
        
        # Get ranked list of CVs
        ranked_cvs = rank_cvs_with_similarity(job_description, cv_texts)
        
        filtered_results = ranked_cvs[:top_n]
        
        for idx, (percentage, similarity, cv_name, cv_text) in enumerate(filtered_results):
            st.subheader(f"Candidate {idx + 1} - Match: {percentage:.2f}%")
            st.text_area(f"CV {idx + 1} - {cv_name}", cv_text, height=200)
            
            # Display the similarity percentage in a more visual format
            st.progress(float(percentage) / 100)  # Ensure it's a valid float and normalized to [0, 1]
            
            # Option to download the selected CV
            st.download_button(
                label="Download CV",
                data=uploaded_files[idx].getvalue(),
                file_name=f"candidate_{idx+1}_cv.{uploaded_files[idx].name.split('.')[-1]}",
                mime="application/octet-stream"
            )

# Footer information with styling
st.markdown("""
    <style>
    .footer {
        text-align: center;
        font-size: 12px;
        color: gray;
    }
    </style>
    <div class="footer">
        Powered by Hugging Face and Streamlit
    </div>
    """, unsafe_allow_html=True)
