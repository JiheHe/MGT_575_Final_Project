# AI News Buddy — MGT 575 Final Project

**Personalized generative-AI news desk:** turn user-supplied interests into grounded story (searched news articles) summaries, editorial visuals, an anchor script, narration audio, and a simple storyboard video — with an explicit multi-step agent trace for transparency.

This README is written so a grader or teammate can **set up the environment, inspect the code, and run the demo** in line with the course final-project expectations.

---

## Repository Link

- **GitHub:** `https://github.com/JiheHe/MGT_575_Final_Project`

---

## What the application does

**Inputs:** comma-separated interests, news region, number of final stories, candidate pool size, persona, tone (main page).

**Outputs:** ranked candidate table, per-story dual summaries (content vs interest angles), optional images, broadcast script, WAV narration, downloadable MP4 storyboard video, and an **agent trace** for debugging.

**Generative AI angle:** Google Gemini (text, image, TTS) with **model chains** in `.env` (the client tries the next model in the chain when a call fails). **Ranking** can fall back to a **deterministic** word-boundary score on titles/snippets when the API is missing or the ranker errors. **Summarization** requires Gemini for full stories: if the API is unavailable it returns **no summaries**; when the API is available but JSON fields are thin, the summarizer fills gaps with **extractive** helpers on the article/snippet text. **Visual prompts**, **images**, **script**, and **voice** each have their own offline or placeholder paths (see below).

**Fallback summary (no API vs API errors):**

| Stage | If `GEMINI_API_KEY` missing / client unavailable | If Gemini is called but fails for a request |
|-------|--------------------------------------------------|---------------------------------------------|
| **Interest context** | Rule-based blurbs + RSS hint heuristics (`src/agents/interest_context.py`) | Same offline path if the enrich call errors |
| **Ranking** | Word-boundary match scores on title+snippet (`src/agents/ranker.py`) | Falls back to the same boundary scorer after an LLM exception |
| **Summaries** | **None** (empty list; no template summaries) | Per-story: skip on hard errors; partial JSON gets **extractive** bullet/why-it-matters fill-ins from article/snippet (`src/utils/text_utils.py`) |
| **Script** | Error-style `BroadcastScript` message (not a generated script) | Requires API once summaries exist |
| **Visual prompts** | Template prompt from headline (`VisualPromptWriterAgent._fallback`) | Same fallback per story if `_with_gemini` raises |
| **Images** | Branded **placeholder** PNG | Same if the image model chain exhausts |
| **Voice** | `None` (no audio file) | `None` if the TTS chain fails |
| **Video** | Skipped when images/audio missing | Trace notes failure / missing inputs |

---

## User interface (Streamlit)

The app is **stepwise** (not a single “generate all” button):

| Step | Button | Role |
|------|--------|------|
| 1 | **Find News** | Profile interests → interest “bundle” context → live RSS (Google + Bing) → rank → **readability / full-text fetch** → grounded **Summarizer** outputs. Clears prior session media. |
| 2 | **Persona** / **Tone** | Used for **Draft Visuals**, **Prepare Script**, and **Record Briefings** — **not** injected into RSS retrieval, ranking, or summarization prompts (those stay interest- and evidence-grounded). |
| 3 | **Draft Visuals** | Image prompts + generation (persona/tone affect prompt style). |
| 4 | **Prepare Script** | Anchor script from structured summaries (persona/tone in instructions). Tight pacing for narration + storyboard is **on** in the UI pipeline. |
| 5 | **Record Briefings** | TTS from script, then storyboard video aligned to audio. |

Buttons use **session state** (locked / ready / done) and disable while a step is running. **Persona** and **Tone** sit on their own row; short explanations live in **`?` tooltips** and a **popover** next to “Generate content”.

---

## System architecture

High-level flow (orchestrator + agents):

```text
InterestProfilerAgent
    → InterestContextAgent (combined lens + RSS query hint)
    → NewsRetrieverAgent (Google News RSS + Bing News RSS, merged/deduped)
    → RankingAgent (Gemini semantic scores on metadata; boundary-match fallback if no API)
    → Readability pass (fetch article text; walk ranked list until enough readable articles)
    → SummarizerAgent (dual summaries; grounded on title/snippet/article text; interests + lens only)
    → [UI] ScriptWriterAgent | VisualPromptWriterAgent + ImageGeneratorAgent
    → [UI] VoiceGeneratorAgent + VideoGeneratorAgent
```

**Orchestrator:** `NewsBuddyOrchestrator` in `src/orchestrator.py` exposes `gather_summaries_only` (step 1), then `generate_story_images`, `generate_anchor_script`, `generate_narration_and_storyboard_video` for the UI. A monolithic `run()` still chains the full pipeline for scripted use.

**Media workspace:** `src/utils/media_workspace.py` clears `generated/images`, `generated/audio`, `generated/video` on a new **Find News** session, clears images before re-drafting visuals, audio before re-recording, and sweeps obvious **MoviePy temp** files from the project root if they appear.

---

## Reproducing the demo

From the **repository root** (`MGT_575_Final_Project/`):

1. **Python 3.11+** recommended (match your course environment).

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Secrets (never commit real keys):**

   ```bash
   copy .env.example .env
   ```

   Edit `.env` and set at least `GEMINI_API_KEY` for full LLM ranking, summarization, script, images, and TTS. Without it, ranking uses a keyword-style fallback and summarization returns no Gemini summaries (the app will surface empty-story warnings).

5. **Run the UI:**

   ```bash
   streamlit run app.py
   ```

6. In the browser: run **Find News** first, then **Draft Visuals** / **Prepare Script** in either order, then **Record Briefings** when both script and images exist.

---

## Configuration (`.env`)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for full Gemini features. |
| `GEMINI_TEXT_MODELS` | Comma-separated text model chain (see `.env.example`). Legacy: `GEMINI_TEXT_MODEL` + `GEMINI_TEXT_FALLBACK_MODELS`. |
| `GEMINI_IMAGE_MODELS` | Image / image-capable model chain. Legacy: `GEMINI_IMAGE_MODEL` + `GEMINI_IMAGE_FALLBACK_MODELS`. |
| `GEMINI_VOICE_MODELS` | TTS model chain. Legacy: `GEMINI_VOICE_MODEL` + `GEMINI_VOICE_FALLBACK_MODELS`. |
| `SEARCH_API_KEY` | Reserved for future search providers; RSS path does not require it. |
| `SEARCH_PROVIDER` | Default `rss` (live RSS retrieval in `NewsRetrieverAgent`). |

Resolved paths and model routing live in `src/config.py` and `src/gemini_client.py`.

---

## Project layout

```text
MGT_575_Final_Project/
  app.py                 # Streamlit UI, step buttons, session state
  requirements.txt
  .env.example           # Template only — copy to .env
  README.md
  src/
    orchestrator.py      # Pipeline coordination
    models.py            # Pydantic data models
    config.py            # Env + generated/ directories
    gemini_client.py     # Gemini text / image / TTS with fallbacks
    agents/              # One module per agent
    utils/               # media_workspace, article_fetcher, text/display helpers
  generated/
    images/ .gitkeep
    audio/  .gitkeep
    video/  .gitkeep     # intermediates; final MP4 may also be offered as bytes in UI
```

---

## Safety, grounding, and ethics

- Summaries are instructed to stay **grounded** in fetched title, snippet, and article body when available; **uncertainty** is surfaced when text is thin.
- **Images** are labeled as AI-generated editorial visuals, not documentary evidence.
- **Persona/tone** intentionally do **not** steer the **summarization** step, to reduce “spin” on source material; they shape **delivery** (script, voice, prompt style) only after facts are condensed.
- **API keys** belong only in `.env` (and in your report: describe variables, not pasted secrets), per standard practice and the course reminder on secrets.

---

## Known limitations

- RSS feeds and HTML fetchers hit **paywalls, bot protection, and sparse pages**; the readability pass may skip many candidates, so the **candidate pool size ≠ number of final stories**.
- **Bing** RSS in this MVP depends on the model-produced **query hint** from `InterestContextAgent`; if that is empty, Bing is skipped.
- **No multi-user persistence**; state is per browser session (`st.session_state`).
- **Video** assembly uses MoviePy; some environments may write temp files — the workspace helper attempts to remove stray root artifacts after **Find News**.

---
