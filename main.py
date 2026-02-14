import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from transformers import pipeline, set_seed

app = FastAPI(title="DistilGPT2 Chat")

# Load the model once at startup (CPU-only)
generator = pipeline(
    "text-generation",
    model="distilbert/distilgpt2",  # DistilGPT2 model on Hugging Face [web:71][web:74]
    device=-1                       # CPU
)
set_seed(42)

@app.get("/")
async def chat(ask: str = Query(..., min_length=1, max_length=200)):
    try:
        output = generator(
            ask,
            max_length=min(len(ask.split()) + 40, 80),
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"question": ask, "answer": output})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
