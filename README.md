# pdf-rag-app

A command-line RAG (Retrieval-Augmented Generation) app that answers
questions grounded in your own PDF documents, using free local embeddings,
a local Chroma vector database, and the Google AI Studio (Gemini) API.

Instead of relying on the model's internal training knowledge, this tool
retrieves the most relevant chunks from your own documents at query time
and asks Gemini to answer using only that retrieved context — reducing
hallucination and making answers traceable back to a specific
document/page.

## How it works

1. **Load** — extracts text from every PDF in a local `docs/` folder
2. **Chunk** — splits page text into overlapping chunks so retrieval can
   find specific relevant passages, not whole documents
3. **Embed** — converts each chunk into a vector using a free, local
   embedding model (`sentence-transformers/all-MiniLM-L6-v2` — no API
   cost, runs entirely on your machine)
4. **Store** — saves all chunk vectors in a local Chroma vector database
5. **Retrieve + Answer** — for each question, finds the most similar
   chunks and sends them to Gemini with a system prompt that restricts it
   to answering only from that context, citing the source document(s)

## Setup

1. Clone the repo and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your-real-key-here
   ```
4. Create a `docs/` folder and add your own PDFs (3–5 works well — try a
   resume, an article, and a report):
   ```bash
   mkdir docs
   # then copy your PDFs into docs/
   ```

## How to run it

```bash
python3 rag_app.py
```

On first run, the embedding model downloads automatically (a few hundred
MB, one-time). Once the vector store is built, you'll get an interactive
prompt:

```
Ready. Ask questions about your documents (type 'exit' to quit).

Q: What is this document about?
```

Example output:
```
Q: What is the role of AI in fintech?

A: Based on the provided document, AI's role in fintech includes driving
technological advancement alongside blockchain and data analytics to
improve efficiency, transparency, and customer experience, and is
expected to be central to future innovation alongside deeper financial
inclusion.
Sources: report.pdf (page 19), report.pdf (page 11)
```

If a question can't be answered from your documents, the app says so
instead of guessing:
```
Q: What is the capital of France?

A: I don't know based on the provided documents.
```

## Configuration

A few constants near the top of `rag_app.py` control retrieval behavior:

| Constant | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks, to avoid cutting ideas mid-sentence |
| `TOP_K` | 5 | How many chunks are retrieved per question |

## Limitations

- **Retrieval quality is uneven across document styles.** Short,
  keyword-heavy documents (like resumes) can retrieve worse against
  conversational questions than prose documents (reports, articles) do,
  because embeddings capture semantic meaning and dense bullet points
  don't "read" like natural sentences. Testing showed that a low `TOP_K`
  (e.g. 3) sometimes caused valid questions about short documents to
  fail with a false "I don't know," because relevant chunks were
  crowded out by longer, more content-dense documents scoring similarly
  in the embedding space. Raising `TOP_K` to 5 fixed most of these
  misses — but this is a real product tradeoff (too low = incomplete
  answers, too high = noisier, slower, costlier answers), not a
  one-time fix.
- **No hybrid search.** Retrieval relies purely on dense vector
  similarity. Adding keyword-based search (e.g. BM25) alongside
  embeddings would likely improve retrieval for structured/short
  documents like resumes.
- **The Gemini API occasionally returns a transient `503 UNAVAILABLE`
  error under high demand.** A simple retry resolves it; a production
  version of this tool would want built-in retry/backoff logic rather
  than relying on the user to re-run manually.
- **No persistent conversation memory** — each question is answered
  independently; the app doesn't remember earlier questions in the same
  session.
