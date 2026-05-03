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

AVAILABLE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-3-27b-it:free",
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
# Agent Function
# =========================

def ask_agent(
    selected_model: str,
    agent_instructions: str,
    project_prompt: str,
    temperature: float = 0.5,
) -> str:
    """
    Sends a request to OpenRouter using the OpenAI-compatible SDK.
    """

    if client is None:
        raise ValueError("OPENROUTER_API_KEY is missing. Please add it to your .env file.")

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": agent_instructions,
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
# Agent Instructions
# =========================

AGENTS = {
    "CEO Analysis": {
        "temperature": 0.5,
        "instructions": """
You are the CEO of an AI software agency.

Your job is to:
1. Evaluate the business value of the project.
2. Identify the target market.
3. Explain the revenue opportunity.
4. Identify business risks.
5. Recommend whether the project is viable.
6. Give a clear executive summary.

Write in a practical, business-focused style.
Use headings, bullet points, and clear recommendations.
""",
    },
    "CTO Technical Plan": {
        "temperature": 0.4,
        "instructions": """
You are the CTO of an AI software agency.

Your job is to:
1. Recommend the best technical architecture.
2. Suggest the technology stack.
3. Identify technical risks.
4. Explain scalability requirements.
5. Give a practical technical build plan.

Write in a way that is useful for a beginner developer and a business client.
Avoid unnecessary complexity.
""",
    },
    "Product Manager Roadmap": {
        "temperature": 0.4,
        "instructions": """
You are the Product Manager of an AI software agency.

Your job is to:
1. Define the target users.
2. List core product features.
3. Prioritize MVP features.
4. Create a realistic product roadmap.
5. Explain product-market fit.
6. Identify what should be delayed until later.

Be practical and focus on what should be built first.
""",
    },
    "Developer Implementation Plan": {
        "temperature": 0.3,
        "instructions": """
You are the Lead Developer of an AI software agency.

Your job is to:
1. Break the project into development tasks.
2. Recommend frontend, backend, database, and API choices.
3. Estimate development effort.
4. Identify implementation challenges.
5. Suggest a practical step-by-step build plan.

Write for a beginner developer who needs clear implementation guidance.
""",
    },
    "Client Success Strategy": {
        "temperature": 0.5,
        "instructions": """
You are the Client Success Manager of an AI software agency.

Your job is to:
1. Explain project expectations to the client.
2. Create a communication plan.
3. Identify client-side risks.
4. Recommend launch and support strategy.
5. Suggest a go-to-market plan.

Write in a professional but beginner-friendly style.
Focus on client satisfaction and project delivery.
""",
    },
}


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
    temperature_adjustment = -0.1
elif temperature_mode == "More Creative":
    temperature_adjustment = 0.2
else:
    temperature_adjustment = 0.0

st.sidebar.markdown("---")
st.sidebar.write("Current model:")
st.sidebar.code(selected_model)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses OpenRouter through the OpenAI-compatible Python SDK."
)


# =========================
# Main UI
# =========================

st.title("🤖 AI Services Agency")
st.write(
    "Enter a project idea and let an AI agency team analyze it from business, technical, product, development, and client-success angles."
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

    results = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_agents = len(AGENTS)

    for index, (agent_name, agent_config) in enumerate(AGENTS.items(), start=1):
        status_text.write(f"Running {agent_name}...")

        try:
            base_temperature = agent_config["temperature"]
            final_temperature = max(
                0.0,
                min(1.0, base_temperature + temperature_adjustment),
            )

            result = ask_agent(
                selected_model=selected_model,
                agent_instructions=agent_config["instructions"],
                project_prompt=project_prompt,
                temperature=final_temperature,
            )

            results[agent_name] = result

        except Exception as e:
            results[agent_name] = f"""
Error while generating {agent_name}.

Details:
{str(e)}

Possible fixes:
1. Check your OpenRouter API key.
2. Try another free model from the sidebar.
3. Wait and try again if the free model is rate-limited.
4. Check your internet connection.
"""

        progress_bar.progress(index / total_agents)

    status_text.success("Analysis complete.")

    tabs = st.tabs(list(results.keys()))

    for tab, agent_name in zip(tabs, results.keys()):
        with tab:
            st.subheader(agent_name)
            st.markdown(results[agent_name])

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

"""

    for agent_name, content in results.items():
        full_report += f"""## {agent_name}

{content}

---

"""

    st.markdown("---")

    st.download_button(
        label="Download Full Report",
        data=full_report,
        file_name="agency_report.md",
        mime="text/markdown",
    )

    st.success("Your AI agency report is ready.")