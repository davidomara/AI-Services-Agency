# AI Services Agency Prototype

## Features

* Project idea analysis
* OpenRouter API integration
* Free model testing and ranking
* Automatic model scoring
* Best-model selection
* Full agency-style report generation
* Markdown report download
* Support for models that do not accept system prompts
* Streamlit web interface
* Environment-based API key management

---

## Example Use Case

Example project input:

```text
Project Name:
Campus Food Delivery App

Project Type:
Web Application

Budget Range:
$10k-$25k

Timeline:
3-4 months

Target Users:
University students, campus restaurants, small food vendors

Main Business Goal:
Help students order food faster and help vendors get more sales.

Project Description:
A mobile app that helps university students order affordable meals from nearby restaurants and campus vendors.

Problem Solved:
Students waste time walking around campus looking for food, and vendors struggle to reach students digitally.

Must-Have Features:
Restaurant listing, menu browsing, ordering, mobile money payment, delivery tracking, vendor dashboard.
```

The app generates a structured report covering:

```text
CEO Analysis
CTO Technical Plan
Product Manager Roadmap
Developer Implementation Plan
Client Success Strategy
```

---

## Tech Stack

* Python
* Streamlit
* OpenRouter
* OpenAI-compatible Python SDK
* python-dotenv

---

## Project Structure

```text
AI-Services-Agency/
├── agency.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
└── .env
```

---

## Requirements

Create a `requirements.txt` file with:

```txt
streamlit
openai
python-dotenv
requests
pydantic
```

For Streamlit Cloud deployment, also create a `runtime.txt` file:

```txt
python-3.12
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-services-agency.git
cd ai-services-agency
```

### 2. Create a virtual environment

On Windows:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a file named `.env` in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

Do not share this file publicly.

### 5. Run the app

```bash
streamlit run agency.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

---

## OpenRouter Models Tested

The app supports benchmarking multiple OpenRouter models.

Recommended default:

```text
nvidia/nemotron-3-super-120b-a12b:free
```

Other models that can be tested:

```text
qwen/qwen3-coder:free
openrouter/free
minimax/minimax-m2.5:free
meta-llama/llama-3.3-70b-instruct:free
google/gemma-3-27b-it:free
google/gemma-3-4b-it:free
```

Free models may be rate-limited depending on provider availability.

---

## Model Benchmarking

The app includes a benchmark mode that:

1. Sends a short test prompt to each selected model.
2. Measures response speed.
3. Scores the output quality.
4. Ranks the models.
5. Saves the best working model in session state.

The scoring considers:

* Response detail
* Business relevance
* Technical usefulness
* MVP advice
* Risk analysis
* Budget and timeline awareness
* Project-specific reasoning
* Response speed

---

## Deployment on Streamlit Community Cloud

### 1. Push the project to GitHub

Make sure `.env` and `venv/` are not committed.

Your `.gitignore` should include:

```gitignore
venv/
.env
__pycache__/
*.pyc
.streamlit/secrets.toml
```

Then push:

```bash
git add .
git commit -m "Initial AI services agency app"
git push
```

### 2. Deploy on Streamlit

Go to:

```text
https://share.streamlit.io
```

Then:

1. Sign in with GitHub.
2. Click **New app**.
3. Select your repository.
4. Set the main file path to:

```text
agency.py
```

5. Click **Deploy**.

### 3. Add Streamlit secrets

In the Streamlit app settings, add:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
```

Do not upload `.env` to GitHub.

---

## Important Deployment Notes

Do not use a full `pip freeze` requirements file from Windows.

Avoid packages like:

```txt
pywin32
```

Streamlit Cloud runs on Linux, so Windows-only packages can break deployment.

Use a clean `requirements.txt`:

```txt
streamlit
openai
python-dotenv
requests
pydantic
```

---

## How It Works

The app uses OpenRouter through the OpenAI-compatible Python SDK:

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
```

The app sends project details to the selected model and asks it to generate a structured agency report.

For some models, especially Google Gemma models, the app combines system instructions and user prompts into a single user message because those models may reject system/developer instructions.

---

## Security

Never commit API keys.

Keep secrets in:

```text
Local development: .env
Streamlit Cloud: App secrets
```

The `.env` file should always be ignored by Git.

---

## Future Improvements

Possible improvements:

* Add user authentication
* Save previous reports
* Export reports as PDF
* Add client branding
* Add cost estimation tables
* Add model provider filtering
* Add retry logic for rate-limited models
* Add a database for report history
* Add custom report templates
* Add PDF upload for project briefs

---

## Author

Built by David Omara as an AI Services Agency prototype using Streamlit and OpenRouter.
