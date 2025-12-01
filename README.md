# Dual‑Model GGUF Chatbot Repository

## Project Overview
This repository contains a **dual‑model chatbot** built with **Gradio** that loads two GGUF LLMs (e.g., an online base model and a custom‑trained model). Given a user‑provided topic, the two models converse back‑and‑forth, allowing you to compare their behaviours (e.g., a mock presidential debate).

## Repository Structure
```
├─ model/                # Original notebook and related files
├─ chatbot/              # Gradio UI and app code
│   ├─ app.py            # Dual‑model chat implementation
│   ├─ README.md         # Instructions for the chatbot app
│   └─ requirements.txt  # Python dependencies
└─ README.md            # **This file** – overall project description
```

## How to Run Locally
```bash
# From the repository root
cd chatbot
pip install -r requirements.txt   # install dependencies
python app.py                       # launch Gradio UI (http://0.0.0.0:7860)
```
Make sure a compatible Llama.cpp server is running on `http://localhost:8000/v1/chat/completions` and can load the GGUF models via the `model` field.

## Deploying to Hugging Face Spaces
1. **Create a new Space** on Hugging Face (choose *Gradio* as the SDK).
2. **Connect the Space to this GitHub repository** (or push the repo manually via `git push` inside the Space).
3. The Space will automatically install the `requirements.txt` and run `app.py`.
4. After the build finishes, you will get a public URL like `https://<your‑username>.hf.space/`.

## Deploying to Streamlit Cloud (alternative)
1. Convert the Gradio UI to a Streamlit app or keep the Gradio app and launch it from a simple `streamlit run app.py` wrapper.
2. Push the repository to GitHub.
3. Sign in to **Streamlit Cloud**, click *New app*, select the repo and branch, and set the main file to `chatbot/app.py`.
4. Streamlit Cloud will install dependencies and provide a public URL.

## Deliverables Checklist
- [x] Source code pushed to a **GitHub repository**.
- [x] Project description for Task 2 placed in the **root `README.md`** (this file).
- [x] Public UI URL hosted on **Hugging Face Spaces** (or Streamlit Cloud) and shared with the reviewer.

Feel free to customize the README further or add additional documentation as needed.
