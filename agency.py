import os
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Nord AI Agency",
    page_icon="🤖",
    layout="wide",
)


# =========================
# Config
# =========================

load_dotenv()


def get_config_value(key: str, default: str | None = None) -> str | None:
    """
    Streamlit Cloud: reads from app secrets.
    Local development: reads from .env.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


OPENROUTER_API_KEY = get_config_value("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = get_config_value(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free",
)


# =========================
# Models
# =========================

AVAILABLE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "openrouter/free",
]

RECOMMENDED_SAFE_DEFAULTS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "openrouter/free",
]


# =========================
# OpenRouter Client
# =========================

def get_openrouter_client() -> OpenAI | None:
    if not OPENROUTER_API_KEY:
        return None

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=45.0,
        max_retries=0,
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
    Google/Gemma routes may reject system/developer instructions.
    For those models, combine instructions and user request into one user message.
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def make_project_prompt(
    client_name: str,
    client_email: str,
    company_name: str,
    industry: str,
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
Client Details:

Client Name:
{client_name}

Client Email:
{client_email}

Company Name:
{company_name}

Industry:
{industry}

Project Details:

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
"""


def user_friendly_error(error_text: str) -> str:
    lower = error_text.lower()

    if "429" in error_text or "rate limit" in lower or "temporarily rate-limited" in lower:
        return "Rate limited"

    if "402" in error_text or "spend limit" in lower:
        return "Spend limit reached"

    if "401" in error_text or "unauthorized" in lower or "api key" in lower:
        return "API key issue"

    if "404" in error_text or "no endpoints found" in lower:
        return "No endpoint available"

    if "400" in error_text or "invalid" in lower:
        return "Invalid request/model"

    if "timeout" in lower or "timed out" in lower:
        return "Timed out"

    return "Failed"


# =========================
# Report Sections
# =========================

REPORT_SECTIONS = [
    {
        "title": "CEO Analysis",
        "max_tokens": 650,
        "prompt": """
Write the CEO Analysis section.

Include:
- Executive summary
- Business value
- Target market
- Revenue opportunity
- Business risks
- Viability recommendation

Be practical, direct, and realistic.
""",
    },
    {
        "title": "CTO Technical Plan",
        "max_tokens": 750,
        "prompt": """
Write the CTO Technical Plan section.

Include:
- Recommended architecture
- Suggested technology stack
- Technical risks
- Scalability requirements
- Practical technical build plan

Make it useful for a beginner developer and a business client.
""",
    },
    {
        "title": "Product Manager Roadmap",
        "max_tokens": 700,
        "prompt": """
Write the Product Manager Roadmap section.

Include:
- Target users
- Core product features
- MVP features
- Features to delay
- Realistic roadmap
- Product-market fit analysis

Focus on what should be built first.
""",
    },
    {
        "title": "Developer Implementation Plan",
        "max_tokens": 750,
        "prompt": """
Write the Developer Implementation Plan section.

Include:
- Frontend plan
- Backend plan
- Database plan
- API/payment plan
- Step-by-step development tasks
- Implementation risks

Keep it practical and implementation-focused.
""",
    },
    {
        "title": "Client Success Strategy",
        "max_tokens": 650,
        "prompt": """
Write the Client Success Strategy section.

Include:
- Client expectations
- Communication plan
- Client-side risks
- Launch strategy
- Support strategy
- Go-to-market plan

Focus on delivery, launch, and client satisfaction.
""",
    },
]


# =========================
# OpenRouter Generation
# =========================

def generate_section_with_fallback(
    section_title: str,
    section_prompt: str,
    project_prompt: str,
    fallback_models: list[str],
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, list[dict]]:
    """
    Tries models in order until one produces the section.

    Returns:
    - section text
    - model used
    - attempt logs
    """

    if client is None:
        raise ValueError("OPENROUTER_API_KEY is missing.")

    system_prompt = f"""
You are Nord AI Agency.

You are writing one section of a client project report.

Section to write:
{section_title}

Rules:
- Output only this section.
- Start with the heading: # {section_title}
- Be practical and beginner-friendly.
- Use clear Markdown.
- Focus on realistic delivery within the stated budget and timeline.
- Be honest about risks and tradeoffs.
- Recommend an MVP-first approach.
- Use the client details when useful, but do not expose private contact information unnecessarily in analysis text.
- Do not mention that you are an AI model.
"""

    user_prompt = f"""
Client and project details:

{project_prompt}

Task:
{section_prompt}
"""

    attempt_logs = []

    for model_id in fallback_models:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=build_messages_for_model(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""

            if not content.strip():
                raise RuntimeError("Model returned an empty response.")

            attempt_logs.append(
                {
                    "section": section_title,
                    "model": model_id,
                    "status": "success",
                    "error": "",
                }
            )

            return content.strip(), model_id, attempt_logs

        except Exception as e:
            error_text = str(e)

            attempt_logs.append(
                {
                    "section": section_title,
                    "model": model_id,
                    "status": user_friendly_error(error_text),
                    "error": error_text,
                }
            )

            continue

    raise RuntimeError(f"All fallback models failed for section: {section_title}")


def generate_report_with_fallbacks(
    project_prompt: str,
    fallback_models: list[str],
    temperature: float,
) -> tuple[str, list[dict]]:
    """
    Generates the report section by section.

    If one model fails on a section, the next model in fallback_models is used.
    Completed sections are preserved.
    """

    report_parts = []
    all_attempt_logs = []

    progress_bar = st.progress(0)
    status_box = st.empty()

    total_sections = len(REPORT_SECTIONS)

    for index, section in enumerate(REPORT_SECTIONS, start=1):
        section_title = section["title"]
        status_box.write(f"Generating {section_title}...")

        section_text, model_used, logs = generate_section_with_fallback(
            section_title=section_title,
            section_prompt=section["prompt"],
            project_prompt=project_prompt,
            fallback_models=fallback_models,
            temperature=temperature,
            max_tokens=section["max_tokens"],
        )

        all_attempt_logs.extend(logs)

        report_parts.append(
            f"{section_text}\n\n_Section generated with: `{model_used}`_"
        )

        progress_bar.progress(index / total_sections)

    status_box.success("Report generation complete.")

    return "\n\n---\n\n".join(report_parts), all_attempt_logs


# =========================
# PDF Export
# =========================

def escape_pdf_text(text: str) -> str:
    """
    Escapes characters that ReportLab Paragraph treats as XML/HTML.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def clean_markdown_for_pdf(text: str) -> str:
    """
    Small Markdown cleanup for PDF text.
    """
    return (
        text.replace("**", "")
        .replace("__", "")
        .replace("`", "")
    )


def markdown_to_pdf_bytes(
    markdown_text: str,
    title: str = "Nord AI Agency Report",
) -> bytes:
    """
    Converts simple Markdown into a PDF file.

    Supports:
    - # headings
    - ## headings
    - ### headings
    - bullet lines
    - normal paragraphs
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        spaceAfter=16,
        alignment=TA_LEFT,
    )

    heading1_style = ParagraphStyle(
        "CustomHeading1",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceBefore=14,
        spaceAfter=8,
    )

    heading2_style = ParagraphStyle(
        "CustomHeading2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4,
    )

    story = []
    story.append(Paragraph(escape_pdf_text(title), title_style))
    story.append(Spacer(1, 0.15 * inch))

    lines = markdown_text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 0.08 * inch))
            continue

        if line == "---":
            story.append(Spacer(1, 0.18 * inch))
            continue

        line = clean_markdown_for_pdf(line)

        if line.startswith("# "):
            story.append(Paragraph(escape_pdf_text(line[2:].strip()), heading1_style))
            continue

        if line.startswith("## "):
            story.append(Paragraph(escape_pdf_text(line[3:].strip()), heading2_style))
            continue

        if line.startswith("### "):
            story.append(Paragraph(escape_pdf_text(line[4:].strip()), heading2_style))
            continue

        if line.startswith("- "):
            story.append(
                Paragraph("• " + escape_pdf_text(line[2:].strip()), bullet_style)
            )
            continue

        if line.startswith("* "):
            story.append(
                Paragraph("• " + escape_pdf_text(line[2:].strip()), bullet_style)
            )
            continue

        story.append(Paragraph(escape_pdf_text(line), body_style))

    doc.build(story)

    pdf_value = buffer.getvalue()
    buffer.close()

    return pdf_value


# =========================
# UI
# =========================

st.title("🤖 Nord AI Agency")

st.write(
    "Generate a practical agency-style project report. "
    "If one model fails or times out, the app automatically tries the next selected model."
)

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key is missing.")
    st.info(
        "Local: add OPENROUTER_API_KEY to `.env`. "
        "Streamlit Cloud: add OPENROUTER_API_KEY in app secrets."
    )
    st.stop()


with st.sidebar:
    st.title("Settings")

    default_index = 0
    if DEFAULT_MODEL in AVAILABLE_MODELS:
        default_index = AVAILABLE_MODELS.index(DEFAULT_MODEL)

    primary_model = st.selectbox(
        "Primary Model",
        AVAILABLE_MODELS,
        index=default_index,
    )

    fallback_models_selected = st.multiselect(
        "Fallback Order",
        AVAILABLE_MODELS,
        default=RECOMMENDED_SAFE_DEFAULTS,
        help="The app tries these in order if the primary model fails or times out.",
    )

    fallback_models = [primary_model]
    for model in fallback_models_selected:
        if model not in fallback_models:
            fallback_models.append(model)

    report_style = st.selectbox(
        "Report Style",
        ["Balanced", "More Precise", "More Creative"],
        index=0,
    )

    if report_style == "More Precise":
        selected_temperature = 0.3
    elif report_style == "More Creative":
        selected_temperature = 0.7
    else:
        selected_temperature = 0.5

    st.markdown("---")
    st.write("Model order:")
    for i, model in enumerate(fallback_models, start=1):
        st.write(f"{i}. `{model}`")

    st.write("Temperature:")
    st.code(str(selected_temperature))

    st.markdown("---")
    st.caption("Version: v1.1 — Lead Capture")


# =========================
# Form
# =========================

with st.form("project_form"):
    st.subheader("Client Details")

    lead_col1, lead_col2 = st.columns(2)

    with lead_col1:
        client_name = st.text_input(
            "Client Name",
            placeholder="Example: David Omara",
        )

        company_name = st.text_input(
            "Company Name",
            placeholder="Example: Nord AI Agency",
        )

    with lead_col2:
        client_email = st.text_input(
            "Client Email",
            placeholder="Example: client@example.com",
        )

        industry = st.text_input(
            "Industry",
            placeholder="Example: Food delivery, education, fintech, healthcare",
        )

    st.markdown("---")
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
# Generate
# =========================

if submitted:
    if not client_name.strip():
        st.error("Please enter the client name.")
        st.stop()

    if not client_email.strip():
        st.error("Please enter the client email.")
        st.stop()

    if "@" not in client_email or "." not in client_email:
        st.error("Please enter a valid client email address.")
        st.stop()

    if not company_name.strip():
        st.error("Please enter the company name.")
        st.stop()

    if not industry.strip():
        st.error("Please enter the industry.")
        st.stop()

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

    if not fallback_models:
        st.error("Please select at least one model.")
        st.stop()

    project_prompt = make_project_prompt(
        client_name=client_name,
        client_email=client_email,
        company_name=company_name,
        industry=industry,
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

    with st.spinner("Generating report..."):
        try:
            ai_report, attempt_logs = generate_report_with_fallbacks(
                project_prompt=project_prompt,
                fallback_models=fallback_models,
                temperature=selected_temperature,
            )
        except Exception as e:
            st.error(
                "All selected models failed. Please try again later or choose different models."
            )
            with st.expander("Technical error details"):
                st.code(str(e))
            st.stop()

    st.success("Report generated successfully.")
    st.markdown(ai_report)

    with st.expander("Model attempt log"):
        for log in attempt_logs:
            status = log["status"]
            section = log["section"]
            model = log["model"]

            if status == "success":
                st.success(f"{section}: {model}")
            else:
                st.warning(f"{section}: {model} failed — {status}")

                if log["error"]:
                    st.code(log["error"])

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_order_text = "\n".join(
        [f"{index}. {model}" for index, model in enumerate(fallback_models, start=1)]
    )

    full_report = f"""# Nord AI Agency Report

Generated: {report_date}

Version: v1.1 — Lead Capture

---

## Client Details

**Client Name:** {client_name}

**Client Email:** {client_email}

**Company Name:** {company_name}

**Industry:** {industry}

---

## Model Order

{model_order_text}

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

    pdf_report = markdown_to_pdf_bytes(
        markdown_text=full_report,
        title=f"Nord AI Agency Report - {project_name}",
    )

    st.markdown("---")

    col_download_1, col_download_2 = st.columns(2)

    with col_download_1:
        st.download_button(
            label="Download Markdown Report",
            data=full_report,
            file_name="nord_ai_agency_report.md",
            mime="text/markdown",
        )

    with col_download_2:
        st.download_button(
            label="Download PDF Report",
            data=pdf_report,
            file_name="nord_ai_agency_report.pdf",
            mime="application/pdf",
        )