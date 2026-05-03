# AI News Buddy — MGT 575 Final Project

**Personalized generative-AI news desk:** turn user-supplied interests into grounded story summaries, editorial visuals, an anchor script, narration audio, and a downloadable storyboard video — with an explicit multi-step **agent trace** for transparency.

- **GitHub:** `https://github.com/JiheHe/MGT_575_Final_Project`
- **Course:** MGT 575 — Generative AI and Social Media (final project)

---

## What the application does

- **Inputs (sidebar):** comma-separated interests, region (RSS edition), number of final stories (1–6), candidate pool size (5–50, default 25). **(Main page):** persona, tone.
- **Outputs:** ranked candidate table, per-story **dual summaries** (Content tab + Interest tab), one **synthetic editorial image** per story, an anchor **broadcast script**, **WAV** narration, an **MP4 storyboard** with per-segment headline overlay, and an **agent trace** of every intermediate decision.
- **Generative AI angle:** Google Gemini for text, image (Imagen / Gemini-image), and TTS, each as a **model chain** in `.env` so the client tries the next model when one fails. Ranking and summarization use structured JSON; both have deterministic fallbacks (see below).

---

## User interface (Streamlit)

The app is **stepwise** (no single "generate everything" button):

| Step | Button | What it does |
|------|--------|--------------|
| 1 | **Find News** | Profile interests → bundle context (lens) → live RSS (Google + Bing) → rank → **readability / full-text fetch** → grounded **Summarizer** outputs. Clears prior session media. |
| 2 | **Persona** / **Tone** | Used for **Draft Visuals**, **Prepare Script**, **Record Briefings** only — **not** injected into RSS retrieval, ranking, or summarization (those stay interest- and evidence-grounded). |
| 3 | **Draft Visuals** | Builds visual prompts and renders one editorial image per story (or a branded placeholder if the model chain refuses). |
| 4 | **Prepare Script** | Writes the anchor broadcast script from the structured summaries; persona/tone shape delivery only. |
| 5 | **Record Briefings** | TTS (persona-mapped voice) + storyboard MP4 in one pass. |

Buttons are gated by a small state machine (`locked` / `ready` / `done` / `busy`) and disabled while any step is running. Persona/Tone live on their own row between *Find News* and the downstream buttons so they can be changed between regenerations without re-running retrieval. Short explanations live in `?` tooltips and a popover next to **Generate content**.

---

## System architecture

```text
InterestProfilerAgent
  -> InterestContextAgent     # bundle lens + RSS query hint
  -> NewsRetrieverAgent       # Google News RSS + Bing News RSS, merged & deduped
  -> RankingAgent             # Gemini semantic scores; word-boundary fallback
  -> Readability gate         # full-text fetch + LLM judge (is_valid, snake_case reason)
  -> SummarizerAgent          # dual summaries grounded on title/snippet/article text
  -> [UI] ScriptWriterAgent | VisualPromptWriterAgent + ImageGeneratorAgent
  -> [UI] VoiceGeneratorAgent + VideoGeneratorAgent
```

The orchestrator (`src/orchestrator.py`) exposes `gather_summaries_only` (Step 1), `generate_story_images`, `generate_anchor_script`, and `generate_narration_and_storyboard_video` for the UI; a monolithic `run()` chains everything for scripted use. Pydantic models in `src/models.py` carry the typed payload between agents.

**Media workspace:** `src/utils/media_workspace.py` clears `generated/images`, `generated/audio`, `generated/video` on a new *Find News* session, clears images before re-drafting visuals, audio before re-recording, and sweeps any stray temp files (`*TEMP_MPY*`, `*TEMP_wvf*`, `briefing_*TEMP*`) from the project root as a precaution.

---

## Fallback behavior (no API key vs. mid-run errors)

Summaries require Gemini and **never** fall back to a templated stand-in — if the LLM is unavailable the story is omitted. The other stages degrade gracefully:

- **Interest context, ranking, visual prompts, script** — each has a deterministic offline path (rule-based bundle blurbs; `\b<term>\b` boundary-match scores on title+snippet+topic; template prompts; a clear "script not generated" message). A successful Gemini call with a thin payload still gets patched with extractive helpers (`fallback_key_points`).
- **Images** — branded placeholder card via Pillow when the image-model chain refuses or errors.
- **Voice / video** — return `None` if their inputs (script / images + audio) aren't both present; the trace records the reason.

The full per-stage decisions are visible in the **Agent Trace** expander at the bottom of the UI.

---

## Reproducing the demo

From the **repository root** (`MGT_575_Final_Project/`):

1. Python 3.11+ recommended.

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Secrets (never commit real keys):

   ```bash
   copy .env.example .env
   ```

   Edit `.env` and set at least `GEMINI_API_KEY` for full Gemini features. Without it, ranking falls back to keyword scoring and summaries are not produced.

5. Run the UI:

   ```bash
   streamlit run app.py
   ```

6. In the browser: **Find News** → **Draft Visuals** / **Prepare Script** (either order) → **Record Briefings**.

---

## Configuration (`.env`)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for Gemini text, image, and TTS. |
| `GEMINI_TEXT_MODELS` | Comma-separated text-model chain. Legacy: `GEMINI_TEXT_MODEL` + `GEMINI_TEXT_FALLBACK_MODELS`. |
| `GEMINI_IMAGE_MODELS` | Image / image-capable model chain. Legacy: `GEMINI_IMAGE_MODEL` + `GEMINI_IMAGE_FALLBACK_MODELS`. |
| `GEMINI_VOICE_MODELS` | TTS model chain. Legacy: `GEMINI_VOICE_MODEL` + `GEMINI_VOICE_FALLBACK_MODELS`. |
| `SEARCH_API_KEY` | Reserved for future paid search providers; RSS path does not require it. |
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
    gemini_client.py     # Gemini text / image / TTS with model chains
    agents/              # One module per agent
    utils/               # media_workspace, article_fetcher, text/display helpers
  generated/
    images/  audio/  video/   # session artifacts; cleared on Find News
                              # (MP4 currently returned in-memory; folder kept
                              #  for legacy/debug paths)
```

---

## Safety, grounding, and ethics

- Summaries stay **grounded** in the fetched title, snippet, and full article body, with a per-story `uncertainty_or_limitations` field rendered verbatim.
- Every generated image is captioned **"AI-generated image (synthetic). Not a real photograph or evidence; used for visual context only."**; visual prompts ban real-person likeness, recognizable public figures, logos, brand names, text overlays, and evidence-style photography.
- **Persona/tone** intentionally do **not** steer summarization or ranking. They shape delivery (script, voice, image style) only after facts are condensed, so a "Friendly buddy / energetic" delivery cannot rephrase a fact into a punchier-but-misleading version of itself.
- API keys belong only in `.env` (and in your report: describe variables, not pasted secrets).

---

## Known limitations (MVP)

- **Retrieval is a black box.** The user has no UI control over date range (Google query is internally scoped to the last 7 days), no publisher whitelist/blacklist, and no per-article approve/reject. This is the most consistent piece of feedback from informal user testing.
- **Pool ≠ final stories.** RSS feeds and HTML fetchers hit paywalls, bot protection, and sparse pages; the readability gate may drop many candidates, so the final number of summaries is often less than the candidate pool size.
- **Bing RSS depends on the LLM query hint** from `InterestContextAgent`. If the hint is empty (transient Gemini outage), Bing is skipped on purpose to avoid a generic Bing query polluting candidates.
- **Image moderation is broad.** Imagen / Gemini-image endpoints occasionally refuse safe editorial prompts; the placeholder card prevents a broken UI but a re-run is sometimes needed to get an actual image.
- **No multi-user persistence.** State is per browser session (`st.session_state`).
- **Video assembly uses `ffmpeg` directly** (located via `imageio-ffmpeg`); MoviePy is only used to read audio duration. The MP4 is built into the OS temp directory, returned in-memory to Streamlit, and unlinked.

---
