# DistilGPT2 Chat on Render

Simple FastAPI wrapper around `distilbert/distilgpt2` with a `/?ask=` query parameter.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
