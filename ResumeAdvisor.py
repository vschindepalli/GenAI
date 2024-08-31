import os
import crewai
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_anthropic import ChatAnthropic
import PyPDF2
import io

# Initialize the ChatAnthropic model
llm = ChatAnthropic(model="claude-3-sonnet-20240229", api_key = "your_api_key")

# Initialize the search tool
search_tool = DuckDuckGoSearchRun()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Define agents
resume_parser = Agent(
    role='Resume Parser',
    goal='Accurately extract and summarize key information from resumes',
    backstory='You are an expert in analyzing resumes and extracting relevant information.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

job_analyzer = Agent(
    role='Job Analyzer',
    goal='Analyze job descriptions and identify key requirements and desired skills',
    backstory='You are an expert in understanding job market trends and employer needs.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool]
)

resume_tailor = Agent(
    role='Resume Tailor',
    goal='Customize resumes to match specific job requirements',
    backstory='You are skilled at highlighting relevant experiences and skills to match job descriptions.',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

cover_letter_writer = Agent(
    role='Cover Letter Writer',
    goal='Create compelling cover letters tailored to specific jobs and candidates',
    backstory='You are an expert in crafting persuasive and personalized cover letters.',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Main function to run the crew
def run_resume_tailoring_crew(pdf_resume, job_description):
    # Extract text from PDF resume
    resume_text = extract_text_from_pdf(pdf_resume)

    # Define tasks
    parse_resume = Task(
    description=f"Parse and summarize the following resume:\n\n{resume_text}",
    agent=resume_parser,
    expected_output="A concise summary of the candidate's experience, skills, and education extracted from the resume."
    )

    analyze_job = Task(
        description=f"Analyze the following job description and identify key requirements and desired skills:\n\n{job_description}",
        agent=job_analyzer,
        expected_output="A list of key requirements (hard and soft skills, experience) and desired qualifications extracted from the job description."
    )

    tailor_resume = Task(
        description="Tailor the resume based on the resume summary and job analysis provided by your crew members.",
        agent=resume_tailor,
        expected_output="A revised resume tailored to highlight the candidate's qualifications that are most relevant to the job description."
    )   

    write_cover_letter = Task(
        description="Write a cover letter based on the resume summary and job analysis provided by your crew members.",
        agent=cover_letter_writer,
        expected_output="A compelling cover letter tailored to the specific job and highlighting the candidate's most relevant qualifications."
    )


    result = crew.kickoff()

    return {
        "resume_summary": result[0],
        "job_analysis": result[1],
        "tailored_resume": result[2],
        "cover_letter": result[3]
    }

# Example usage
if __name__ == "__main__":
    # Load PDF resume
    with open("VishalSai_Chindepalli_Freshman_Resume_General.pdf", "rb") as pdf_file:
        pdf_resume = pdf_file.read()

    job_description = """
    We are seeking a skilled Software Engineer with experience in cloud technologies.
    Required skills:
    - Strong proficiency in Python and JavaScript
    - Experience with AWS or other cloud platforms
    - Knowledge of React and Node.js
    - Excellent problem-solving skills
    """

    results = run_resume_tailoring_crew(pdf_resume, job_description)
    
    print("Resume Summary:")
    print(results["resume_summary"])
    print("\nJob Analysis:")
    print(results["job_analysis"])
    print("\nTailored Resume:")
    print(results["tailored_resume"])
    print("\nCover Letter:")
    print(results["cover_letter"])