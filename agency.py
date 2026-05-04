import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


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


APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "assets" / "nord_logo.png"

OPENROUTER_API_KEY = get_config_value("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = get_config_value(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free",
)

EMAIL_HOST = get_config_value("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(get_config_value("EMAIL_PORT", "465"))
EMAIL_USER = get_config_value("EMAIL_USER")
EMAIL_PASSWORD = get_config_value("EMAIL_PASSWORD")
EMAIL_TO = get_config_value("EMAIL_TO", "nordlink256@gmail.com")


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
# Session State
# =========================

if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""

if "full_report" not in st.session_state:
    st.session_state.full_report = ""

if "pdf_report" not in st.session_state:
    st.session_state.pdf_report = None

if "attempt_logs" not in st.session_state:
    st.session_state.attempt_logs = []

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

if "last_project_name" not in st.session_state:
    st.session_state.last_project_name = "nord_ai_agency_report"

if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

if "email_message" not in st.session_state:
    st.session_state.email_message = ""


# =========================
# Prompt Helpers
# =========================

def build_messages_for_model(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
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
- Use Markdown tables when useful.
- Focus on realistic delivery within the stated budget and timeline.
- Be honest about risks and tradeoffs.
- Recommend an MVP-first approach.
- Use only standard ASCII punctuation where possible.
- Avoid special dashes, smart quotes, box symbols, unusual bullets, and emojis.
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
# Email Helper
# =========================

def send_report_email(
    client_name: str,
    client_email: str,
    company_name: str,
    project_name: str,
    markdown_report: str,
    pdf_report: bytes,
) -> tuple[bool, str]:
    """
    Sends the generated report to Nord with Markdown and PDF attachments.
    """

    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_TO:
        return (
            False,
            "Email settings are missing. Check EMAIL_USER, EMAIL_PASSWORD, and EMAIL_TO.",
        )

    safe_project_name = (
        project_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    subject = f"New Nord AI Agency Report - {project_name}"

    body = f"""
A new Nord AI Agency report has been generated.

Client Name: {client_name}
Client Email: {client_email}
Company Name: {company_name}
Project Name: {project_name}

Attached:
1. Markdown report
2. PDF report
"""

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    msg.add_attachment(
        markdown_report.encode("utf-8"),
        maintype="text",
        subtype="markdown",
        filename=f"{safe_project_name}_nord_ai_agency_report.md",
    )

    msg.add_attachment(
        pdf_report,
        maintype="application",
        subtype="pdf",
        filename=f"{safe_project_name}_nord_ai_agency_report.pdf",
    )

    try:
        with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        return True, f"Report emailed to {EMAIL_TO}."

    except Exception as e:
        return False, str(e)


# =========================
# PDF Export Helpers
# =========================

def normalize_text_for_pdf(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u202f": " ",
        "\u00a0": " ",
        "\ufeff": "",
        "\u25a0": "-",
        "\u25cf": "-",
        "\u2022": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2248": "~",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2192": "->",
        "\u2190": "<-",
        "\u00d7": "x",
        "\u2713": "Yes",
        "\u2705": "Yes",
        "\u274c": "No",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return "".join(ch if ord(ch) < 128 else "-" for ch in text)


def escape_pdf_text(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def clean_markdown_for_pdf(text: str) -> str:
    return (
        text.replace("**", "")
        .replace("__", "")
        .replace("`", "")
    )


def is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False

    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False

    for cell in cells:
        cleaned = cell.replace(":", "").replace("-", "").strip()
        if cleaned:
            return False

    return True


def is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]


def split_markdown_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def collect_markdown_table(lines: list[str], start_index: int) -> tuple[list[list[str]], int]:
    table_rows = []
    index = start_index

    while index < len(lines):
        line = lines[index].strip()

        if not is_markdown_table_row(line):
            break

        if is_markdown_table_separator(line):
            index += 1
            continue

        table_rows.append(split_markdown_table_row(line))
        index += 1

    return table_rows, index


def build_pdf_table(
    rows: list[list[str]],
    available_width: float,
    header_style: ParagraphStyle,
    cell_style: ParagraphStyle,
) -> Table:
    max_cols = max(len(row) for row in rows)

    normalized_rows = []
    for row_index, row in enumerate(rows):
        padded_row = row + [""] * (max_cols - len(row))
        style = header_style if row_index == 0 else cell_style
        normalized_rows.append(
            [
                Paragraph(
                    escape_pdf_text(clean_markdown_for_pdf(cell)),
                    style,
                )
                for cell in padded_row
            ]
        )

    col_width = available_width / max_cols

    table = Table(
        normalized_rows,
        colWidths=[col_width] * max_cols,
        repeatRows=1 if len(normalized_rows) > 1 else 0,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F6FB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111A4D")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2F2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    return table


def add_pdf_footer(canvas, doc):
    canvas.saveState()

    page_number = canvas.getPageNumber()
    footer_text = f"Nord AI Agency | Page {page_number}"

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(
        A4[0] - 0.65 * inch,
        0.35 * inch,
        footer_text,
    )

    canvas.setStrokeColor(colors.HexColor("#D9E2F2"))
    canvas.setLineWidth(0.4)
    canvas.line(
        0.65 * inch,
        0.5 * inch,
        A4[0] - 0.65 * inch,
        0.5 * inch,
    )

    canvas.restoreState()


def markdown_to_pdf_bytes(
    markdown_text: str,
    title: str = "Nord AI Agency Report",
    logo_path: Path | None = None,
    client_name: str = "",
    company_name: str = "",
    project_name: str = "",
    report_date: str = "",
) -> bytes:
    markdown_text = normalize_text_for_pdf(markdown_text)

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
    )

    page_width, _ = A4
    available_width = page_width - doc.leftMargin - doc.rightMargin

    styles = getSampleStyleSheet()

    brand_blue = colors.HexColor("#0057B8")
    brand_navy = colors.HexColor("#111A4D")
    brand_light = colors.HexColor("#F3F6FB")
    brand_border = colors.HexColor("#D9E2F2")
    brand_dark = colors.HexColor("#111111")
    muted_text = colors.HexColor("#555555")

    title_style = ParagraphStyle(
        "NordTitle",
        parent=styles["Title"],
        fontSize=21,
        leading=26,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor=brand_blue,
    )

    subtitle_style = ParagraphStyle(
        "NordSubtitle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
        textColor=brand_dark,
    )

    heading1_style = ParagraphStyle(
        "NordHeading1",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        spaceBefore=14,
        spaceAfter=7,
        textColor=brand_navy,
    )

    heading2_style = ParagraphStyle(
        "NordHeading2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=9,
        spaceAfter=5,
        textColor=brand_blue,
    )

    body_style = ParagraphStyle(
        "NordBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        spaceAfter=5,
        textColor=brand_dark,
    )

    bullet_style = ParagraphStyle(
        "NordBullet",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        leftIndent=16,
        firstLineIndent=-8,
        spaceAfter=4,
        textColor=brand_dark,
    )

    small_style = ParagraphStyle(
        "NordSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        spaceAfter=4,
        textColor=muted_text,
    )

    table_header_style = ParagraphStyle(
        "NordTableHeader",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9,
        textColor=brand_navy,
        fontName="Helvetica-Bold",
    )

    table_cell_style = ParagraphStyle(
        "NordTableCell",
        parent=styles["BodyText"],
        fontSize=7.2,
        leading=9,
        textColor=brand_dark,
    )

    story = []

    logo_cell = ""
    if logo_path and logo_path.exists():
        logo_cell = Image(str(logo_path), width=1.55 * inch, height=1.05 * inch)

    header_table = Table(
        [
            [
                logo_cell,
                Paragraph(
                    "<b>Nord AI Agency</b><br/>Project Strategy Report",
                    title_style,
                ),
            ]
        ],
        colWidths=[1.85 * inch, 4.75 * inch],
    )

    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LINEBELOW", (0, 0), (-1, -1), 1.25, brand_blue),
            ]
        )
    )

    story.append(header_table)
    story.append(Spacer(1, 0.18 * inch))

    cover_data = [
        ["Project", project_name or "N/A"],
        ["Client", client_name or "N/A"],
        ["Company", company_name or "N/A"],
        ["Generated", report_date or "N/A"],
    ]

    cover_table = Table(
        cover_data,
        colWidths=[1.35 * inch, 5.1 * inch],
        hAlign="LEFT",
    )

    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), brand_light),
                ("TEXTCOLOR", (0, 0), (0, -1), brand_navy),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 0.35, brand_border),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(cover_table)
    story.append(Spacer(1, 0.22 * inch))

    story.append(
        Paragraph(
            "This report provides a practical business, technical, product, development, and client-success analysis for the proposed project.",
            subtitle_style,
        )
    )

    story.append(Spacer(1, 0.12 * inch))

    lines = markdown_text.splitlines()

    skip_sections = {
        "Generated:",
        "Version:",
    }

    skip_headings = {
        "## Client Details",
        "## Model Order",
    }

    skip_until_next_hr = False
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        if line in skip_headings:
            skip_until_next_hr = True
            i += 1
            continue

        if skip_until_next_hr:
            if line == "---":
                skip_until_next_hr = False
            i += 1
            continue

        if any(line.startswith(prefix) for prefix in skip_sections):
            i += 1
            continue

        if not line:
            story.append(Spacer(1, 0.045 * inch))
            i += 1
            continue

        if line == "---":
            story.append(Spacer(1, 0.12 * inch))
            i += 1
            continue

        if (
            is_markdown_table_row(line)
            and i + 1 < len(lines)
            and is_markdown_table_separator(lines[i + 1])
        ):
            table_rows, next_index = collect_markdown_table(lines, i)

            if table_rows:
                pdf_table = build_pdf_table(
                    rows=table_rows,
                    available_width=available_width,
                    header_style=table_header_style,
                    cell_style=table_cell_style,
                )
                story.append(Spacer(1, 0.06 * inch))
                story.append(pdf_table)
                story.append(Spacer(1, 0.08 * inch))

            i = next_index
            continue

        line = clean_markdown_for_pdf(line)

        if line.startswith("# Nord AI Agency Report"):
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(escape_pdf_text(line[2:].strip()), heading1_style))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Paragraph(escape_pdf_text(line[3:].strip()), heading2_style))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(escape_pdf_text(line[4:].strip()), heading2_style))
            i += 1
            continue

        if line.startswith("- "):
            story.append(
                Paragraph("- " + escape_pdf_text(line[2:].strip()), bullet_style)
            )
            i += 1
            continue

        if line.startswith("* "):
            story.append(
                Paragraph("- " + escape_pdf_text(line[2:].strip()), bullet_style)
            )
            i += 1
            continue

        if line.startswith("_Section generated with:"):
            story.append(Paragraph(escape_pdf_text(line), small_style))
            i += 1
            continue

        story.append(Paragraph(escape_pdf_text(line), body_style))
        i += 1

    doc.build(
        story,
        onFirstPage=add_pdf_footer,
        onLaterPages=add_pdf_footer,
    )

    pdf_value = buffer.getvalue()
    buffer.close()

    return pdf_value


# =========================
# UI Header
# =========================

header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=170)
    else:
        st.markdown("## 🤖")

with header_col2:
    st.title("Nord AI Agency")
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


# =========================
# Sidebar
# =========================

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)

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
    st.caption("Version: v1.2 - Email Reports")


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
            placeholder="Example: Nord Projects",
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

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_order_text = "\n".join(
        [f"{index}. {model}" for index, model in enumerate(fallback_models, start=1)]
    )

    full_report = f"""# Nord AI Agency Report

Generated: {report_date}

Version: v1.2 - Email Reports

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

    full_report_for_pdf = normalize_text_for_pdf(full_report)

    pdf_report = markdown_to_pdf_bytes(
        markdown_text=full_report_for_pdf,
        title=f"Nord AI Agency Report - {project_name}",
        logo_path=LOGO_PATH,
        client_name=client_name,
        company_name=company_name,
        project_name=project_name,
        report_date=report_date,
    )

    st.session_state.ai_report = ai_report
    st.session_state.full_report = full_report
    st.session_state.pdf_report = pdf_report
    st.session_state.attempt_logs = attempt_logs
    st.session_state.report_generated = True
    st.session_state.last_project_name = project_name

    email_sent, email_message = send_report_email(
        client_name=client_name,
        client_email=client_email,
        company_name=company_name,
        project_name=project_name,
        markdown_report=full_report,
        pdf_report=pdf_report,
    )

    st.session_state.email_sent = email_sent
    st.session_state.email_message = email_message

    st.success("Report generated successfully.")


# =========================
# Persistent Report Display
# =========================

if st.session_state.report_generated and st.session_state.full_report:
    st.markdown("---")
    st.subheader("Generated Report")

    if st.session_state.email_message:
        if st.session_state.email_sent:
            st.success(st.session_state.email_message)
        else:
            st.warning(
                f"Report generated, but email sending failed: {st.session_state.email_message}"
            )

    st.markdown(st.session_state.ai_report)

    with st.expander("Model attempt log"):
        for log in st.session_state.attempt_logs:
            status = log["status"]
            section = log["section"]
            model = log["model"]

            if status == "success":
                st.success(f"{section}: {model}")
            else:
                st.warning(f"{section}: {model} failed - {status}")

                if log["error"]:
                    st.code(log["error"])

    st.markdown("---")

    col_download_1, col_download_2, col_clear = st.columns(3)

    safe_project_name = (
        st.session_state.last_project_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    with col_download_1:
        st.download_button(
            label="Download Markdown Report",
            data=st.session_state.full_report,
            file_name=f"{safe_project_name}_nord_ai_agency_report.md",
            mime="text/markdown",
            key="download_markdown_report",
        )

    with col_download_2:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.pdf_report,
            file_name=f"{safe_project_name}_nord_ai_agency_report.pdf",
            mime="application/pdf",
            key="download_pdf_report",
        )

    with col_clear:
        if st.button("Clear Report"):
            st.session_state.ai_report = ""
            st.session_state.full_report = ""
            st.session_state.pdf_report = None
            st.session_state.attempt_logs = []
            st.session_state.report_generated = False
            st.session_state.last_project_name = "nord_ai_agency_report"
            st.session_state.email_sent = False
            st.session_state.email_message = ""
            st.rerun()