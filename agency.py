import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Load Environment Variables
# =========================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =========================
# Available OpenRouter Models
# =========================
# Keep Nemotron first because it already worked for you.
# Free models can still be rate-limited depending on OpenRouter/provider load.

AVAILABLE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
    "minimax/minimax-m2.5:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-4b-it:free",
]


# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title="AI Services Agency",
    page_icon="🤖",
    layout="wide",
)


# =========================
# OpenRouter Client
# =========================

def get_openrouter_client():
    if not OPENROUTER_API_KEY:
        return None

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


client = get_openrouter_client()


# =========================
# Report Generator
# =========================

def generate_full_report(
    selected_model: str,
    project_prompt: str,
    temperature: float = 0.5,
) -> str:
    """
    Generates the full agency report in ONE OpenRouter request.

    This is better for free OpenRouter models because it avoids making
    5 separate API calls for CEO, CTO, Product Manager, Developer,
    and Client Success Manager.
    """

    if client is None:
        raise ValueError("OPENROUTER_API_KEY is missing. Please add it to your .env file.")

    system_prompt = """
You are an AI Services Agency made of five expert roles:

1. CEO
2. CTO
3. Product Manager
4. Lead Developer
5. Client Success Manager

Generate ONE complete agency report.

Use these exact Markdown sections:

# CEO Analysis
Include:
- Executive summary
- Business value
- Target market
- Revenue opportunity
- Business risks
- Viability recommendation

# CTO Technical Plan
Include:
- Recommended architecture
- Suggested technology stack
- Technical risks
- Scalability requirements
- Practical technical build plan

# Product Manager Roadmap
Include:
- Target users
- Core product features
- MVP features
- Features to delay
- Realistic roadmap
- Product-market fit analysis

# Developer Implementation Plan
Include:
- Frontend plan
- Backend plan
- Database plan
- API/payment plan
- Step-by-step development tasks
- Implementation risks

# Client Success Strategy
Include:
- Client expectations
- Communication plan
- Client-side risks
- Launch strategy
- Support strategy
- Go-to-market plan

Important rules:
- Be practical and beginner-friendly.
- Focus on realistic delivery within the budget and timeline.
- Be honest about tradeoffs.
- Recommend an MVP-first approach.
- Use headings, bullet points, and tables where useful.
- Do not include generic hype.
- Do not mention that you are an AI model.
"""

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": project_prompt,
            },
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content


# =========================
# Sidebar
# =========================

st.sidebar.title("⚙️ Settings")

default_index = 0
if DEFAULT_MODEL in AVAILABLE_MODELS:
    default_index = AVAILABLE_MODELS.index(DEFAULT_MODEL)

selected_model = st.sidebar.selectbox(
    "Choose OpenRouter Model",
    AVAILABLE_MODELS,
    index=default_index,
)

temperature_mode = st.sidebar.selectbox(
    "Creativity Level",
    ["Balanced", "More Precise", "More Creative"],
    index=0,
)

if temperature_mode == "More Precise":
    selected_temperature = 0.3
elif temperature_mode == "More Creative":
    selected_temperature = 0.7
else:
    selected_temperature = 0.5

st.sidebar.markdown("---")
st.sidebar.write("Current model:")
st.sidebar.code(selected_model)

st.sidebar.write("Temperature:")
st.sidebar.code(str(selected_temperature))

st.sidebar.markdown("---")
st.sidebar.info(
    "This version uses one OpenRouter call to reduce free-model rate-limit errors."
)


# =========================
# Main UI
# =========================

st.title("🤖 AI Services Agency")

st.write(
    "Enter a project idea and generate a full agency-style report with CEO, CTO, Product Manager, Developer, and Client Success sections."
)

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is missing. Please add it to your `.env` file before running the app."
    )

    st.code(
        """
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
""",
        language="env",
    )

    st.stop()


# =========================
# Project Input Form
# =========================

with st.form("project_form"):
    st.subheader("Project Details")

    col1, col2 = st.columns(2)

    with col1:
        project_name = st.text_input(
            "Project Name",
            placeholder="Example: Campus Food Delivery App",
        )

        project_type = st.selectbox(
            "Project Type",
            [
                "Web Application",
                "Mobile App",
                "Progressive Web App",
                "API Development",
                "Data Analytics",
                "AI/ML Solution",
                "Automation System",
                "Other",
            ],
        )

        budget_range = st.selectbox(
            "Budget Range",
            [
                "$1k-$5k",
                "$5k-$10k",
                "$10k-$25k",
                "$25k-$50k",
                "$50k-$100k",
                "$100k+",
            ],
            index=2,
        )

    with col2:
        target_users = st.text_input(
            "Target Users",
            placeholder="Example: University students and campus vendors",
        )

        timeline = st.selectbox(
            "Timeline",
            [
                "Less than 1 month",
                "1-2 months",
                "3-4 months",
                "5-6 months",
                "6+ months",
            ],
            index=2,
        )

        business_goal = st.text_input(
            "Main Business Goal",
            placeholder="Example: Help students order food faster and help vendors get more sales",
        )

    project_description = st.text_area(
        "Project Description",
        placeholder="Describe the project idea in detail...",
        height=120,
    )

    main_problem = st.text_area(
        "Problem the Project Solves",
        placeholder="What pain point does this project solve?",
        height=100,
    )

    must_have_features = st.text_area(
        "Must-Have Features",
        placeholder="Example: Restaurant listing, menu browsing, ordering, mobile money payment, delivery tracking, vendor dashboard",
        height=100,
    )

    submitted = st.form_submit_button("Analyze Project")


# =========================
# Run Analysis
# =========================

if submitted:
    if not project_name.strip():
        st.error("Please enter the project name.")
        st.stop()

    if not project_description.strip():
        st.error("Please enter the project description.")
        st.stop()

    if not main_problem.strip():
        st.error("Please enter the problem this project solves.")
        st.stop()

    if not must_have_features.strip():
        st.error("Please enter the must-have features.")
        st.stop()

    project_prompt = f"""
Analyze this client project.

Project Name: {project_name}
Project Type: {project_type}
Budget Range: {budget_range}
Timeline: {timeline}

Target Users:
{target_users}

Main Business Goal:
{business_goal}

Project Description:
{project_description}

Problem Solved:
{main_problem}

Must-Have Features:
{must_have_features}

Instructions:
- Give practical, beginner-friendly advice.
- Focus on realistic delivery within the budget and timeline.
- Be honest about risks and tradeoffs.
- Recommend an MVP-first approach.
- Format the answer clearly with headings and bullet points.
"""

    st.markdown("---")
    st.subheader("Analysis Results")

    with st.spinner("Generating full agency report..."):
        try:
            ai_report = generate_full_report(
                selected_model=selected_model,
                project_prompt=project_prompt,
                temperature=selected_temperature,
            )

        except Exception as e:
            st.error("Something went wrong while generating the report.")
            st.code(str(e))

            st.warning(
                "Try switching back to `nvidia/nemotron-3-super-120b-a12b:free`, because that model already worked in your earlier test."
            )

            st.stop()

    st.success("Analysis complete.")

    st.markdown(ai_report)

    # =========================
    # Build Downloadable Report
    # =========================

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_report = f"""# AI Services Agency Report

Generated: {report_date}

Model Used: {selected_model}

---

## Project Details

**Project Name:** {project_name}

**Project Type:** {project_type}

**Budget Range:** {budget_range}

**Timeline:** {timeline}

**Target Users:** {target_users}

**Main Business Goal:** {business_goal}

---

## Project Description

{project_description}

---

## Problem Solved

{main_problem}

---

## Must-Have Features

{must_have_features}

---

{ai_report}
"""

    st.markdown("---")

    st.download_button(
        label="Download Full Report",
        data=full_report,
        file_name="agency_report.md",
        mime="text/markdown",
    )

    st.success("Your AI agency report is ready.")