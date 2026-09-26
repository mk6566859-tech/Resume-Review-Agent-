# Resume Review Studio

A beginner-friendly, single-agent resume review app built with Streamlit, CrewAI, and Groq. It compares resume evidence with a target job description and produces a structured review without inventing qualifications or experience.

## Brief architecture explanation

The app has one CrewAI **Agent**, one **Task**, and one **Crew**. The Streamlit page accepts resume text or a PDF and a job description. `pypdf` extracts text from PDFs locally; CrewAI sends the extracted text and job description to the selected Groq model. The result is shown as a structured Markdown review and can be downloaded.

The reviewer treats both inputs as evidence, not instructions. It marks unsupported or ambiguous requirements as **Unknown / Not Demonstrated** and never recommends claiming a keyword unless it is true.

## Project file structure

```text
resume-review-agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

`config.toml` sets the dark, warm-brown color theme. `secrets.toml.example` is a template only; the real secrets file is private and ignored by Git.

## Requirements

- Python 3.12
- A Groq API key
- GitHub account (for Community Cloud deployment)

Dependencies are pinned in `requirements.txt`:

- `streamlit==1.32.2`
- `crewai==0.80.0`
- `crewai-tools==0.17.0`
- `embedchain==0.1.114`
- `litellm==1.55.12`
- `pypdf==4.1.0`
- `setuptools==80.9.0`
- `reportlab==4.2.5`

The companion-package pins keep CrewAI's transitive dependencies compatible with the requested `pypdf==4.1.0`. `setuptools` is pinned because CrewAI 0.80 imports `pkg_resources`, which was removed from newer setuptools releases. LiteLLM provides CrewAI's Groq integration and the app's API error types. The default model is `openai/gpt-oss-120b`. Groq model availability can change: check the Groq Console's production model list before deploying, and set `GROQ_MODEL` to an active model if needed.

## Run locally

1. Install Python 3.12 and Git.
2. Open a terminal in this project folder and create a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

3. Install the pinned dependencies:

   ```text
   pip install -r requirements.txt
   ```

4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Replace the example value with your real key from the Groq Console. Never commit or share this real file.
5. Start the app:

   ```text
   streamlit run app.py
   ```

6. Open the local address shown in the terminal, provide a resume and job description, and choose **Generate my review**.

## Add the project to GitHub

1. Create a new, empty GitHub repository.
2. Put the files from this folder into the repository, including `.streamlit/config.toml`.
3. Do **not** upload `.streamlit/secrets.toml`. `.gitignore` excludes it; commit only the example template.
4. Commit and push the project:

   ```text
   git init
   git add app.py requirements.txt README.md .gitignore .streamlit/config.toml .streamlit/secrets.toml.example
   git commit -m "Build evidence-based resume review app"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```

Replace the GitHub URL with your own repository address.

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and authorize access to the repository.
2. Choose **Create app**, select the repository, branch (usually `main`), and main file path `app.py`.
3. In the deployment's **Advanced settings**, select **Python 3.12**.
4. Add the secrets in the deployment's **Secrets** field using TOML syntax:

   ```toml
   GROQ_API_KEY = "your-real-groq-api-key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```

   Enter the real key in the Cloud secrets interface only. Do not add it to GitHub, `app.py`, or `secrets.toml.example`.

5. Deploy the app. If the model is no longer listed as a production model in Groq, change `GROQ_MODEL` to a currently active Groq model and restart/redeploy.

## Privacy and limitations

- PDF text is extracted on the app server in memory; the PDF file itself is not sent to Groq.
- Resume text and job-description text are sent to the configured Groq model for analysis.
- The app does not write input files or text to its own database or disk. Streamlit keeps the page's current values during the active session.
- Avoid submitting data you are not comfortable sharing with Groq. Check the provider's current privacy and retention terms.
- Scanned PDFs with no selectable text need OCR before they can be reviewed; this minimal app does not include OCR.

## Common errors and fixes

| Message or symptom | What to check |
| --- | --- |
| `GROQ_API_KEY is missing` | Add the key to `.streamlit/secrets.toml` locally or the Cloud app's Secrets setting. Restart the app after changing it. |
| PDF could not be read | Confirm it is a valid, uncorrupted PDF and is no larger than 10 MB. |
| No selectable text found | The PDF may be a scan or image. Paste the resume text or use a PDF with selectable text. |
| Password-protected PDF | Remove the password and upload an unlocked copy. |
| Groq authentication error | Check that the key is correct, active, and entered as `GROQ_API_KEY`. |
| Groq rate limit | Wait briefly and try again. You may have reached your account's request or token limit. |
| Request timed out | Retry when the network and Groq service are available; shorter inputs can also help. |
| Model error | Check that `GROQ_MODEL` names an active production model available to your account. |
| Package installation fails | Confirm the Cloud deployment uses Python 3.12 and that `requirements.txt` is present at the repository root. |
| Windows asks for C++ Build Tools during install | CrewAI 0.80 pulls an older Chroma native extension without a Python 3.12 Windows wheel. Install Microsoft C++ Build Tools to run this pinned stack locally on Windows, or use the Linux-based Community Cloud deployment. |

## How the application works

1. You paste resume text or upload a PDF. For PDFs, `pypdf` reads selectable text locally; no OCR is performed.
2. You paste a job description and submit both inputs.
3. The app checks for missing content, unreadable/empty PDFs, oversize files, and missing secrets.
4. A single CrewAI agent receives both texts with strict instructions to distinguish evidence from unknowns.
5. The app displays the nine review sections and offers a formatted PDF download. API failures are translated into clear messages rather than shown as stack traces.
