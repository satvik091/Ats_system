GLA University ATS System
Overview
The GLA University ATS System is a sophisticated Applicant Tracking System (ATS) developed using Streamlit and Google Gemini Pro API. This system automates resume evaluation against job descriptions, providing detailed insights, including match percentages, keyword analysis, and profile summaries. It helps streamline the recruitment process for talent acquisition professionals.

Features
Resume Match Percentage: Calculate how closely a resume matches a job description.
Keyword Analysis: Identify relevant and non-relevant skills in resumes.
Plagiarism Detection: Check resumes for plagiarism against the job description.
Relevant Projects Extraction: Highlight projects that align with the job description.
Skill Recommendations: Suggest missing skills required for the job.
Detailed Resume Evaluation: Provide a comprehensive professional analysis of resumes.
Tech Stack
Python
Streamlit: For creating an interactive web app.
Google Gemini Pro API: For advanced content evaluation.
PyPDF2: For extracting text from PDFs.
NLTK: For natural language processing and tokenization.
FPDF: For generating PDFs.
Plotly: For data visualization (e.g., progress bars).
Pillow: For image processing.
Installation and Setup
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/gla-ats-system.git
cd gla-ats-system
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Set Up Google Gemini API:

Obtain an API key from Google Makersuite.
Create a .env file in the project directory and add your API key:
env
Copy code
GOOGLE_API_KEY=your_api_key_here
Run the App:

bash
Copy code
streamlit run app.py
Access the Application: Open your browser and navigate to http://localhost:8501.

Usage
Upload Job Description:

Use the sidebar to upload a job description in PDF format.
Upload Resume:

Upload a candidate's resume in PDF format.
Choose an Option:

Select an action from the sidebar, such as:
Percentage Match
Show Relevant Skills
Non-Relevant Skills
Plagiarism Score
Relevant Projects
Recommended Skills
Tell Me About the Resume
View Results:

Results will be displayed dynamically on the interface.
Prompts Used
Percentage Match: Numerical match percentage.
Show Relevant Skills: Lists skills aligning with the job description.
Non-Relevant Skills: Lists skills not relevant to the job.
Plagiarism Score: Indicates the percentage of plagiarized content.
Relevant Projects: Lists projects that fit the job requirements.
Recommended Skills: Suggests missing skills.
Tell Me About the Resume: Provides a professional evaluation summary.
Screenshots

License
This project is licensed under the MIT License.

Contributing
Feel free to fork this repository, create issues, or submit pull requests. Contributions are welcome!

Contact
For any inquiries or issues, contact [Your Name] at [Your Email].

