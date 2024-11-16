# GLA University ATS System

A Streamlit-based Applicant Tracking System (ATS) application, integrated with Google's Gemini API, designed to streamline candidate evaluation. This system evaluates resumes against job descriptions using advanced features like match percentage calculation, keyword analysis, and profile summaries.

## Features

- **Resume Match Percentage**: Calculates how well a resume matches the job description.
- **Keyword Analysis**: Identifies missing and relevant criteria from the resume.
- **Plagiarism Check**: Calculates plagiarism percentage for resumes.
- **Skill Analysis**: Highlights relevant and non-relevant skills based on the job description.
- **Recommended Skills**: Suggests skills that are missing from the resume but are present in the job description.
- **Project Evaluation**: Extracts relevant projects that match the job description.
- **Profile Summary**: Provides a detailed analysis of strengths, weaknesses, and overall fit.

## Getting Started

### Prerequisites

- Python 3.x
- All required Python packages are listed in `requirements.txt`.

### Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gla-ats-system.git
    cd gla-ats-system
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the Google Gemini API**:
    - Obtain an API key from [Google Gemini](https://makersuite.google.com/).
    - Store it in a `.env` file as follows:
      ```
      AI_API_KEY=your_gemini_api_key
      ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

5. **Usage**:
    - Upload a job description in PDF format.
    - Upload a resume in PDF format.
    - Use the sidebar to select options such as "Percentage Match," "Relevant Skills," etc.

### Files

- `app.py`: The main Streamlit application.
- `.env`: Environment variables, including the API key for Google Gemini.
- `requirements.txt`: Contains all the dependencies required for the project.

## How It Works

1. **Resume Analysis**:
    - Extracts text from PDF resumes and job descriptions.
    - Uses prompts to query the Google Gemini API for detailed evaluations.

2. **Gemini Integration**:
    - Leverages the Gemini API for natural language processing and analysis tasks.

3. **Streamlit UI**:
    - Provides a user-friendly interface for uploading files and viewing results.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://deepmind.google/technologies/gemini/#introduction)
