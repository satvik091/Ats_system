
import streamlit as st
import PyPDF2
import nltk
from collections import Counter
from docx import Document
import difflib  # For calculating similarity in plagiarism check
from dotenv import load_dotenv
load_dotenv()
import base64
import os
from PIL import Image
import pdf2image
import google.generativeai as genai
from io import BytesIO
from fpdf import FPDF

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def get_gemini_response1(input_prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(input_prompt)
    return response.text

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


def input_pdf_setup(pdf_file):
    return [extract_text_from_pdf(pdf_file)]

def extract_skills(text):
    skill_set = {'python', 'java', 'data analysis', 'project management', 'machine learning', 'communication', 'sql'}
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    relevant_skills = skill_set.intersection(cleaned_tokens)
    return relevant_skills

def generate_pdf(content):
    # Function to generate a PDF from the given resume content
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Split content into lines and add each line to the PDF
    for line in content.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)

    # Save the PDF to a BytesIO object
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output

def generate_pdf(content):
    # Function to generate a PDF from the given resume content
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Replace unsupported Unicode characters
    content = content.replace("\u2013", "-")  # Replace en-dash with a hyphen
    content = content.replace("\u2014", "--")  # Replace em-dash with double hyphen
    content = content.replace("\u2022", "*")  # Replace bullet points with asterisks

    # Split content into lines and add each line to the PDF
    for line in content.split("\n"):
        try:
            pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
        except UnicodeEncodeError as e:
            st.error(f"Error encoding line: {line} - {e}")

    # Save the PDF to a BytesIO object
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output

def generate_resume():
    st.header("Create Your Resume")

    # Taking inputs from the user
    name = st.text_input("Enter Name")
    email = st.text_input("Enter Email")
    phone = st.text_input("Enter Phone Number")
    skills = st.text_area("Enter Skills")
    education = st.text_area("Enter Education")
    work_experience = st.text_area("Enter Work Experience")
    projects = st.text_area("Enter Projects")
    achievements = st.text_area("Enter Achievements")
    certifications = st.text_area("Enter Certifications")
    hobbies = st.text_area("Enter Hobbies")

    # Once all inputs are received
    if st.button("Generate Resume"):
        # Prepare the input prompt for AI generation
        input_prompt9 = f"""
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
        Your task is to create the resume for the provided job description and return the resume that was created by using the user inputs:
        Name: {name}
        Email: {email}
        Phone Number: {phone}

        Skills: {skills}

        Education: {education}

        Work Experience: {work_experience}

        Projects: {projects}

        Achievements: {achievements}

        Certifications: {certifications}

        Hobbies: {hobbies}
        The output should come as a PDF file containing the generated resume.
        """

        # Get the response from the AI model
        response = get_gemini_response1(input_prompt9)

        # Display the generated resume to the user
        st.subheader("Generated Resume")
        st.write(response)

        # Generate a PDF from the response
        pdf = generate_pdf(response)

        # Provide a download button for the generated PDF
        st.download_button(
            label="Download Resume as PDF",
            data=pdf,
            file_name=f"{name}_resume.pdf",
            mime="application/pdf"
        )




def tailor_resume(resume_text, job_desc_text):
    # Extract skills from the job description
    jd_skills = extract_skills(job_desc_text)

    # Tokenize and clean the resume text
    resume_tokens = word_tokenize(resume_text.lower())
    resume_cleaned_tokens = [word for word in resume_tokens if word.isalpha() and word not in stopwords.words('english')]

    # Identify relevant and missing skills
    relevant_resume_skills = set(resume_cleaned_tokens).intersection(jd_skills)
    missing_skills = jd_skills - relevant_resume_skills

    # Load resume into a docx template for editing
    doc = Document()
    doc.add_heading('Tailored Resume', 0)

    # Split resume text into paragraphs
    resume_lines = resume_text.split("\n")

    # Add the resume content to the docx, while suggesting improvements
    for line in resume_lines:
        doc.add_paragraph(line)

    # Insert a section for adding missing skills
    if missing_skills:
        doc.add_heading('Suggested Improvements (Based on Job Description)', level=1)
        doc.add_paragraph(
            "The following skills are suggested to be added based on the job description:\n" +
            ', '.join(missing_skills) + ".\nPlease consider adding them to your resume, especially in sections like 'Skills' or 'Experience'."
        )

    # Add relevant skills (already in the resume)
    doc.add_heading('Relevant Skills Already Present', level=1)
    doc.add_paragraph(', '.join(relevant_resume_skills))

    # Return the tailored resume as a downloadable DOCX file
    output = BytesIO()
    doc.save(output)
    return output.getvalue()

# Streamlit UI
st.title("GLA University ATS System")
st.subheader("About")
st.write("This sophisticated ATS project, developed with Gemini Pro and Streamlit, seamlessly incorporates advanced features including resume match percentage, keyword analysis to identify missing criteria, and the generation of comprehensive profile summaries, enhancing the efficiency and precision of the candidate evaluation process for discerning talent acquisition professionals.")

st.markdown("""
  - [Streamlit](https://streamlit.io/)
  - [Gemini Pro](https://deepmind.google/technologies/gemini/#introduction)
  - [makersuit API Key](https://makersuite.google.com/)

  """)

# Sidebar for input
st.sidebar.header("Upload Your Job Description")
job_desc_file = st.sidebar.file_uploader("Upload Job Description (PDF)", type="pdf")
op=st.sidebar.selectbox("Resume:",["Choose an option","Yes, I have","No, I have to create."])

#Prompts
input_prompt1 = """
 You are an experienced Technical Human Resource Manager tasked with evaluating resumes against job descriptions. 
Using the provided job description and resume text, perform a detailed analysis. Highlight the following:
1. Candidate's strengths in relation to the job requirements.
2. Key weaknesses or areas needing improvement.
3. Specific skills or experiences that align with the job description.
4. Overall fit for the role.
Return the evaluation in a clear, professional format.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Your task is to evaluate the resume against the provided job description and provide a match percentage.
The output should be a numerical percentage value only, without any additional text or symbols (e.g., 75).
"""

input_prompt4 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality,
your task is to evaluate the resume against the provided job description. give me the relevant skills if the resume matches
the job description. The output should come as text containing all relevant skills required for given job description .
"""

input_prompt5 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality,
your task is to evaluate the resume against the provided job description. give me the non-relevant skills if the resume matches
the job description. The output should come as text containing all non-relevant skills mentioned in resume that are not required for given job description .
"""

input_prompt6 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Your task is to evaluate the resume against the provided job description and return only the plagiarism score, expressed
as a percentage of similarity between the resume with all global resumes.
"""

input_prompt7 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Your task is to evaluate the resume against the provided job description and return only the Relevant Projects, for the given job description.
The output should come as text containing all relevant projects required for given job description.
"""

input_prompt8 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Your task is to evaluate the resume against the provided job description and return only the Recommended Skills required for Job Description, for the given resume.
The output should come as text containing all recommended skills required for given job description.
"""


# If a job description is uploaded
if job_desc_file is not None:
    pdf_content = input_pdf_setup(job_desc_file)
    job_desc_text = pdf_content[0]

    st.subheader("Your Resume")
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")

    if resume_file is not None:
        opt = st.sidebar.selectbox("Available Options", ["Choose an option", "Percentage match", "Show Relevant Skills", "Non-relevant Skills", "Plagiarism Score", "Relevant Projects", "Recommended Skills", "Tell Me About the Resume"])

        resume_pdf_content = input_pdf_setup(resume_file)
        resume_text = resume_pdf_content[0]

        # Get match percentage
        if opt == "Percentage match":
            response = get_gemini_response(input_prompt3, pdf_content, job_desc_text[0])
            # Display the percentage as a progress bar
            st.subheader("Percentage Match")
            st.progress(int(response))
            st.write(f"Match: {response}%")

        # Get relevant skills
        if opt == "Show Relevant Skills":
            relevant_skills = get_gemini_response(resume_text, pdf_content, input_prompt4)
            st.write("Relevant Skills:")
            st.write(relevant_skills)

        # Get non-relevant skills
        if opt == "Non-relevant Skills":
            non_relevant_skills = get_gemini_response(resume_text, pdf_content, input_prompt5)
            st.write("Non-Relevant Skills:")
            st.write(non_relevant_skills)

        # Get plagiarism percentage
        if opt == "Plagiarism Score":
            response = get_gemini_response(input_prompt6, pdf_content, job_desc_text[0])
            st.subheader("Plagiarism Score")
            # Display the percentage as a progress bar
            st.progress(int(response))
            st.write(f"Match: {response}%")

        # Get relevant projects
        if opt == "Relevant Projects":
            relevant_projects = get_gemini_response(resume_text, pdf_content, input_prompt7)
            st.write("Relevant Projects:")
            st.write(relevant_projects)

        # Get recommended skills
        if opt == "Recommended Skills":
            recommended_skills = get_gemini_response(resume_text, pdf_content, input_prompt8)
            st.write("Recommended Skills:")
            st.write(recommended_skills)

        if opt == "Tell Me About the Resume":
            response = get_gemini_response( resume_text, job_desc_text[0], input_prompt1)
            st.subheader("Resume Tells")
            st.write(response)
