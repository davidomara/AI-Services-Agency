import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Nord AI Agency",
    page_icon="🤖",
    layout="wide",
)


# =========================
# Load Environment Variables
# =========================

load_dotenv()


def get_config_value(key: str, default: str | None = None) -> str | None:
    """
    Reads config from Streamlit secrets first, then local .env.

    Streamlit Cloud: use app secrets.
    Local development: use .env.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


OPENROUTER_API_KEY = get_config_value("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_config_value(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free",
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =========================
# OpenRouter Client
# =========================

def get_openrouter_client() -> OpenAI | None:
    if not OPENROUTER_API_KEY:
        return None

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=60.0,
        max_retries=1,
    )


client = get_openrouter_client()


# =========================
# Prompt Helpers
# =========================

def build_messages_for_model(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
    """
    Some models, especially Gemma-style models, may reject system/developer messages.
    This helper keeps the app compatible if you change models later.
    """

    model_lower = model_id.lower()

    if "gemma" in model_lower or model_lower.startswith("google/"):
        return [
            {
                "role": "user",
                "content": f"""
Instructions:
{system_prompt}

User request:
{user_prompt}
""",
            }
        ]

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def make_project_prompt(
    project_name: str,
    project_type: str,
    budget_range: str,
    timeline: str,
    target_users: str,
    business_goal: str,
    project_description: str,
    main_problem: str,
    must_have_features: str,
) -> str:
    return f"""
Analyze this client project.

Project Name:
{project_name}

Project Type:
{project_type}

Budget Range:
{budget_range}

Timeline:
{timeline}

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
- Keep recommendations specific to this project.
"""


# =========================
# Error Handling
# =========================

def user_friendly_error(error_text: str) -> str:
    lower = error_text.lower()

    if "429" in error_text or "rate limit" in lower or "temporarily rate-limited" in lower:
        return (
            "The AI provider is temporarily rate-limited. "
            "Please try again in a few minutes."
        )

    if "402" in error_text or "spend limit" in lower:
        return (
            "The OpenRouter API key has reached its spending limit. "
            "Please check your OpenRouter account settings."
        )

    if "401" in error_text or "api key" in lower or "unauthorized" in lower:
        return (
            "The OpenRouter API key is missing or invalid. "
            "Please check the app secrets."
        )

    if "404" in error_text or "no endpoints found" in lower:
        return (
            "The selected model currently has no available endpoint. "
            "Please update OPENROUTER_MODEL in app secrets."
        )

    if "timeout" in lower:
        return (
            "The request took too long. "
            "Please try again with a shorter project description."
        )

    return "Something went wrong while generating the report."


# =========================
# Report Generator
# =========================

def generate_full_report(
    project_prompt: str,
    temperature: float = 0.5,
) -> str:
    """
    Generates one complete agency report with a single OpenRouter request.
    """

    if client is None:
        raise ValueError("OPENROUTER_API_KEY is missing.")

    system_prompt = """
You are Nord AI Agency, a practical AI services agency made of five expert roles:

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
- Focus on realistic delivery within the stated budget and timeline.
- Be honest about tradeoffs.
- Recommend an MVP-first approach.
- Use headings, bullet points, and simple tables where useful.
- Avoid generic hype.
- Do not mention that you are an AI model.
- Keep the report detailed but not excessively long.
"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=build_messages_for_model(
            model_id=OPENROUTER_MODEL,
            system_prompt=system_prompt,
            user_prompt=project_prompt,
        ),
        temperature=temperature,
        max_tokens=2200,
    )

    return response.choices[0].message.content


# =========================
# Main UI
# =========================

st.title("🤖 Nord AI Agency")

st.write(
    "Generate a practical business, product, technical, and launch report for your project idea."
)

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key is missing.")
    st.info(
        "If running locally, add `OPENROUTER_API_KEY` to your `.env` file. "
        "If deployed on Streamlit Cloud, add it in app secrets."
    )
    st.stop()


with st.sidebar:
    st.title("Settings")
    st.write("Model")
    st.code(OPENROUTER_MODEL)

    selected_temperature_label = st.selectbox(
        "Report Style",
        ["Balanced", "More Precise", "More Creative"],
        index=0,
    )

    if selected_temperature_label == "More Precise":
        selected_temperature = 0.3
    elif selected_temperature_label == "More Creative":
        selected_temperature = 0.7
    else:
        selected_temperature = 0.5

    st.write("Temperature")
    st.code(str(selected_temperature))

    st.markdown("---")
    st.info("Production mode: one model, one report generation request.")


# =========================
# Form
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
        placeholder=(
            "Example: Restaurant listing, menu browsing, ordering, "
            "mobile money payment, delivery tracking, vendor dashboard"
        ),
        height=100,
    )

    submitted = st.form_submit_button("Generate Agency Report")


# =========================
# Generate Report
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

    project_prompt = make_project_prompt(
        project_name=project_name,
        project_type=project_type,
        budget_range=budget_range,
        timeline=timeline,
        target_users=target_users,
        business_goal=business_goal,
        project_description=project_description,
        main_problem=main_problem,
        must_have_features=must_have_features,
    )

    st.markdown("---")
    st.subheader("Agency Report")

    with st.spinner("Generating your agency report..."):
        try:
            ai_report = generate_full_report(
                project_prompt=project_prompt,
                temperature=selected_temperature,
            )

        except Exception as e:
            error_text = str(e)
            st.error(user_friendly_error(error_text))

            with st.expander("Technical error details"):
                st.code(error_text)

            st.stop()

    st.success("Report generated successfully.")
    st.markdown(ai_report)

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_report = f"""# Nord AI Agency Report

Generated: {report_date}

Model Used: {OPENROUTER_MODEL}

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
        label="Download Report",
        data=full_report,
        file_name="nord_ai_agency_report.md",
        mime="text/markdown",
    )