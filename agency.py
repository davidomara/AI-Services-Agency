import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Load Environment Variables
# =========================

load_dotenv()

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))

DEFAULT_MODEL = st.secrets.get(
    "OPENROUTER_MODEL",
    os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =========================
# Available OpenRouter Models
# =========================

AVAILABLE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "openrouter/free",
    "minimax/minimax-m2.5:free",
]

RECOMMENDED_SAFE_DEFAULTS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "openrouter/free",
    "minimax/minimax-m2.5:free",
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
# Prompt Helpers
# =========================

def build_messages_for_model(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
    """
    Some OpenRouter models, especially Google Gemma through Google AI Studio,
    may reject system/developer instructions.

    For those models, combine system instructions and user prompt into one user message.
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


# =========================
# Error + Scoring Helpers
# =========================

def classify_openrouter_error(error_text: str) -> str:
    error_text_lower = error_text.lower()

    if (
        "429" in error_text
        or "rate limit" in error_text_lower
        or "temporarily rate-limited" in error_text_lower
    ):
        return "Rate limited"

    if "developer instruction is not enabled" in error_text_lower:
        return "No system prompt support"

    if "404" in error_text or "no endpoints found" in error_text_lower:
        return "No endpoint"

    if "400" in error_text or "not a valid model id" in error_text_lower:
        return "Invalid model"

    if "402" in error_text or "spend limit" in error_text_lower:
        return "Spend limit"

    if "api key" in error_text_lower:
        return "API key issue"

    return "Failed"


def score_model_response(text: str, elapsed_seconds: float) -> tuple[int, list[str]]:
    """
    Simple local scoring. No extra AI call is used.

    Score rewards:
    - Useful detail
    - Required business/technical sections
    - Concrete MVP advice
    - Risks/tradeoffs
    - Budget/timeline awareness
    - Project-specific reasoning
    """

    score = 0
    reasons = []

    cleaned = text.strip()
    lower = cleaned.lower()
    word_count = len(cleaned.split())

    if word_count >= 250:
        score += 25
        reasons.append("Strong detail")
    elif word_count >= 180:
        score += 20
        reasons.append("Good detail")
    elif word_count >= 100:
        score += 12
        reasons.append("Acceptable detail")
    elif word_count >= 50:
        score += 6
        reasons.append("Short but usable")
    else:
        score -= 15
        reasons.append("Too short")

    section_keywords = [
        "business",
        "technical",
        "mvp",
        "risk",
        "roadmap",
        "budget",
        "timeline",
        "users",
        "revenue",
        "launch",
    ]

    matched_keywords = [keyword for keyword in section_keywords if keyword in lower]
    score += len(matched_keywords) * 5

    if matched_keywords:
        reasons.append(f"Covers {len(matched_keywords)} key areas")

    practical_terms = [
        "first",
        "build",
        "start",
        "validate",
        "pilot",
        "vendor",
        "payment",
        "dashboard",
        "tracking",
        "support",
    ]

    practical_matches = [term for term in practical_terms if term in lower]
    score += min(len(practical_matches) * 3, 24)

    if practical_matches:
        reasons.append("Practical implementation advice")

    if "$10k" in cleaned or "$25k" in cleaned or "3-4" in cleaned or "3–4" in cleaned:
        score += 10
        reasons.append("Uses budget/timeline context")

    if "campus" in lower and "food" in lower:
        score += 8
        reasons.append("Understands project context")

    weak_terms = [
        "i cannot",
        "i can't",
        "unable to",
        "as an ai",
        "error",
    ]

    if any(term in lower for term in weak_terms):
        score -= 10
        reasons.append("Contains weak/error-like wording")

    if elapsed_seconds <= 10:
        score += 8
        reasons.append("Fast response")
    elif elapsed_seconds <= 25:
        score += 4
        reasons.append("Reasonable speed")
    else:
        score -= 5
        reasons.append("Slow response")

    score = max(0, min(100, score))

    if not reasons:
        reasons.append("No strong quality signals")

    return score, reasons


# =========================
# Model Test + Ranking
# =========================

def test_model_with_sample(
    model_id: str,
    project_prompt: str,
    temperature: float = 0.3,
) -> dict:
    """
    Tests one model using a short agency-style prompt.
    Returns status, sample output, score, and timing.
    """

    if client is None:
        raise ValueError("OPENROUTER_API_KEY is missing. Please add it to your .env file.")

    test_system_prompt = """
You are testing whether this model is good for an AI Services Agency app.

Return a concise but useful mini-report with exactly these headings:

# Business
# Technical
# MVP
# Risks
# Recommendation

Keep it practical and specific to the user's project.
"""

    mini_prompt = f"""
Create a short test analysis for this project.

{project_prompt}

Limit the response to 250-400 words.
"""

    started_at = time.time()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=build_messages_for_model(
                model_id=model_id,
                system_prompt=test_system_prompt,
                user_prompt=mini_prompt,
            ),
            temperature=temperature,
            max_tokens=700,
        )

        elapsed = time.time() - started_at
        output = response.choices[0].message.content or ""
        score, reasons = score_model_response(output, elapsed)

        return {
            "model": model_id,
            "status": "Working",
            "score": score,
            "seconds": round(elapsed, 2),
            "reason": ", ".join(reasons),
            "sample": output,
            "error": "",
        }

    except Exception as e:
        elapsed = time.time() - started_at
        error_text = str(e)
        error_type = classify_openrouter_error(error_text)

        return {
            "model": model_id,
            "status": error_type,
            "score": 0,
            "seconds": round(elapsed, 2),
            "reason": error_type,
            "sample": "",
            "error": error_text,
        }


# =========================
# Full Report Generator
# =========================

def generate_full_report(
    selected_model: str,
    project_prompt: str,
    temperature: float = 0.5,
) -> str:
    """
    Generates the full agency report in ONE OpenRouter request.
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
        messages=build_messages_for_model(
            model_id=selected_model,
            system_prompt=system_prompt,
            user_prompt=project_prompt,
        ),
        temperature=temperature,
    )

    return response.choices[0].message.content


# =========================
# Session State
# =========================

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []

if "best_model" not in st.session_state:
    st.session_state.best_model = DEFAULT_MODEL

if "latest_report" not in st.session_state:
    st.session_state.latest_report = ""


# =========================
# Sidebar
# =========================

st.sidebar.title("⚙️ Settings")

default_index = 0
if st.session_state.best_model in AVAILABLE_MODELS:
    default_index = AVAILABLE_MODELS.index(st.session_state.best_model)
elif DEFAULT_MODEL in AVAILABLE_MODELS:
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

st.sidebar.write("Best ranked model:")
st.sidebar.code(st.session_state.best_model)

st.sidebar.write("Temperature:")
st.sidebar.code(str(selected_temperature))

st.sidebar.markdown("---")
st.sidebar.info(
    "Use benchmark mode to test models with a short prompt before generating the full report."
)


# =========================
# Main UI
# =========================

st.title("🤖 AI Services Agency")

st.write(
    "Enter a project idea, benchmark free OpenRouter models, rank their outputs, then generate a full agency-style report with the best model."
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

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        benchmark_submitted = st.form_submit_button("Test and Rank Models")

    with col_b:
        report_submitted = st.form_submit_button("Generate Full Report")

    with col_c:
        best_report_submitted = st.form_submit_button("Use Best Model for Report")


# =========================
# Validation
# =========================

def validate_project_inputs() -> bool:
    if not project_name.strip():
        st.error("Please enter the project name.")
        return False

    if not project_description.strip():
        st.error("Please enter the project description.")
        return False

    if not main_problem.strip():
        st.error("Please enter the problem this project solves.")
        return False

    if not must_have_features.strip():
        st.error("Please enter the must-have features.")
        return False

    return True


# =========================
# Build Prompt
# =========================

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


# =========================
# Benchmark Models
# =========================

if benchmark_submitted:
    if not validate_project_inputs():
        st.stop()

    st.markdown("---")
    st.subheader("Model Benchmark Results")

    st.warning(
        "This will call each selected model once. Free models may still return 429 rate-limit errors."
    )

    selected_models_to_test = st.multiselect(
        "Models to test",
        AVAILABLE_MODELS,
        default=RECOMMENDED_SAFE_DEFAULTS,
    )

    if not selected_models_to_test:
        st.error("Please select at least one model to test.")
        st.stop()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, model_id in enumerate(selected_models_to_test, start=1):
        status_text.write(f"Testing {model_id}...")

        result = test_model_with_sample(
            model_id=model_id,
            project_prompt=project_prompt,
            temperature=0.3,
        )

        results.append(result)
        progress_bar.progress(index / len(selected_models_to_test))

        # Small delay helps avoid burst rate limits.
        time.sleep(1)

    ranked_results = sorted(
        results,
        key=lambda item: (item["score"], -item["seconds"]),
        reverse=True,
    )

    st.session_state.benchmark_results = ranked_results

    working_results = [
        item for item in ranked_results
        if item["status"] == "Working" and item["score"] > 0
    ]

    if working_results:
        st.session_state.best_model = working_results[0]["model"]
        st.success(f"Best working model: {st.session_state.best_model}")
    else:
        st.error("No working model found in this benchmark run.")

    status_text.success("Benchmark complete.")


# =========================
# Display Benchmark Results
# =========================

if st.session_state.benchmark_results:
    st.markdown("---")
    st.subheader("Ranked Models")

    table_rows = []

    for rank, item in enumerate(st.session_state.benchmark_results, start=1):
        table_rows.append(
            {
                "Rank": rank,
                "Model": item["model"],
                "Status": item["status"],
                "Score": item["score"],
                "Seconds": item["seconds"],
                "Reason": item["reason"],
            }
        )

    st.dataframe(table_rows, use_container_width=True)

    st.subheader("Model Samples")

    for rank, item in enumerate(st.session_state.benchmark_results, start=1):
        with st.expander(
            f"#{rank} — {item['model']} — {item['status']} — Score {item['score']}"
        ):
            if item["sample"]:
                st.markdown(item["sample"])

            if item["error"]:
                st.code(item["error"])


# =========================
# Generate Full Report
# =========================

should_generate_report = report_submitted or best_report_submitted

if should_generate_report:
    if not validate_project_inputs():
        st.stop()

    model_for_report = selected_model

    if best_report_submitted:
        model_for_report = st.session_state.best_model

    st.markdown("---")
    st.subheader("Analysis Results")

    st.write("Using model:")
    st.code(model_for_report)

    with st.spinner("Generating full agency report..."):
        try:
            ai_report = generate_full_report(
                selected_model=model_for_report,
                project_prompt=project_prompt,
                temperature=selected_temperature,
            )

        except Exception as e:
            st.error("Something went wrong while generating the report.")
            st.code(str(e))

            st.warning(
                "Run `Test and Rank Models` first, then use the best working model for the full report."
            )

            st.stop()

    st.session_state.latest_report = ai_report

    st.success("Analysis complete.")
    st.markdown(ai_report)

    # =========================
    # Build Downloadable Report
    # =========================

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_report = f"""# AI Services Agency Report

Generated: {report_date}

Model Used: {model_for_report}

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