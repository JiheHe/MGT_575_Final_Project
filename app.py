from __future__ import annotations

import re

import streamlit as st

from src.config import load_config
from src.models import BriefingResult, UserProfile
from src.orchestrator import NewsBuddyOrchestrator
from src.utils.display_text import humanize_display_text, humanize_single_line

_HELP_INTERESTS = (
    "Comma-separated topics. The app reads them as one bundle for search, ranking, and how each story ties back to you."
)
_HELP_REGION = "Region code for news RSS (for example US). Affects which edition of feeds is queried."
_HELP_MAX_STORIES = (
    "Target number of top stories after ranking and readability checks. "
    "Fewer summaries than your candidate pool is normal when many links fail full-text fetch."
)
_HELP_CANDIDATE_POOL = (
    "How many distinct RSS headlines to merge (after dedup) before ranking. "
    "The app then walks ranked items until enough pass a full-text readability check to fill “Number of stories”—"
    "so you often see fewer finished stories than this number, and some URLs never yield a summary."
)
_HELP_FIND_NEWS = (
    "Runs interest profiling, live RSS retrieval, relevance ranking, article-text fetch, and grounded summaries. "
    "Unlocks Draft Visuals and Prepare Script."
)
_HELP_DRAFT_VISUALS = (
    "Builds image prompts and renders story art from your summaries, using the Persona and Tone above. "
    "Requires Find News first; can run before or after Prepare Script."
)
_HELP_PREPARE_SCRIPT = (
    "Writes the anchor broadcast script from summaries using Persona, Tone, and tight pacing tuned for narration + video. "
    "Requires Find News first; can run before or after Draft Visuals."
)
_HELP_RECORD_BRIEFINGS = (
    "One pass: synthesizes narration from your script, then cuts the storyboard video to match. "
    "Requires summaries, anchor script, and story images."
)
_HELP_PERSONA = (
    "Delivery character for the anchor script and TTS, plus style cues in image prompts. "
    "Applies when you run Draft Visuals, Prepare Script, or Record Briefings—change before those steps."
)
_HELP_TONE = (
    "Speaking and writing style for the script, prompt wording, and narration energy. "
    "Applies when you run Draft Visuals, Prepare Script, or Record Briefings—change before those steps."
)
_HELP_GENERATE_POPOVER = """
**Workflow**

- **Find News** — profiles interests, pulls candidates, ranks them, fetches readable article text, writes grounded summaries.
- **Draft Visuals** and **Prepare Script** — either order after Find News; both need summaries loaded.
- **Record Briefings** — voice + storyboard video in one step; needs script and images.

**Why the pool feels “smaller”**

The candidate pool is only the *starting list*. The app keeps trying ranked URLs until enough yield usable full text for your **Number of stories**. Paywalls, bot blocks, and thin pages drop many items before summarization.
"""


def _safe_inline(text: str) -> str:
    """Escape markdown-special chars for inline dynamic strings."""
    t = humanize_single_line(text, max_len=2000)
    # Include $ so dollar amounts are not parsed as math/LaTeX in markdown.
    return re.sub(r"([\\`*_{}\[\]()#+\-!|>$])", r"\\\1", t)


def _escape_markdown_block(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-!|>$])", r"\\\1", text)


def _markdown_block(text: str) -> None:
    """Render prose with markdown escaping for visual consistency."""
    body = humanize_display_text(text).strip()
    if not body:
        st.caption("—")
        return
    st.markdown(_escape_markdown_block(body))


def _uncertainty_plain(text: str) -> None:
    """Plain text so model dash-lists are not parsed as markdown."""
    body = humanize_display_text(text).strip()
    if not body:
        st.caption("—")
        return
    st.text(body)


def _strip_leading_bullet_markers(line: str) -> str:
    """Collapse '- -', '•', etc. so we emit a single markdown bullet."""
    t = line.strip()
    while t:
        if t.startswith("- "):
            t = t[2:].strip()
            continue
        if t.startswith("--"):
            t = t[2:].strip()
            continue
        if t.startswith("-") and len(t) > 1 and not t[1].isdigit():
            t = t[1:].strip()
            continue
        if t.startswith("•") or t.startswith("*"):
            t = t[1:].strip()
            continue
        break
    return t


def _bullet_lines(items: list[str]) -> None:
    chunks: list[str] = []
    for raw in items:
        h = humanize_display_text(raw).strip()
        if not h:
            continue
        if "\n" in h:
            sublines = []
            for part in h.split("\n"):
                p = part.strip()
                if not p:
                    continue
                body = _strip_leading_bullet_markers(p)
                if body:
                    sublines.append(_escape_markdown_block(f"- {body}"))
            chunks.extend(sublines)
            continue
        body = _strip_leading_bullet_markers(h)
        if body:
            chunks.append(_escape_markdown_block(f"- {body}"))
    st.markdown("\n\n".join(chunks) if chunks else "—")


def _profile_with_ui_style(prof: UserProfile) -> UserProfile:
    """Apply main-page Persona/Tone widgets (session keys) for regeneration without re-running Find News."""
    return prof.model_copy(
        update={
            "persona": str(st.session_state.get("buddy_persona", "Lisa - professional anchor")),
            "tone": str(st.session_state.get("buddy_tone", "professional")),
        }
    )


def _merge_result(base: BriefingResult, **updates: object) -> BriefingResult:
    updates_dict = dict(updates)
    trace_add = updates_dict.pop("agent_trace", None)
    merged = base.model_copy(update=updates_dict)
    if trace_add is not None:
        merged = merged.model_copy(update={"agent_trace": list(base.agent_trace) + list(trace_add)})
    return merged


def _render_briefing_result(result: BriefingResult) -> None:
    prof = result.user_profile
    st.subheader("How we're reading your interests")
    st.caption(
        "The app interprets your list as one bundle so ambiguous terms (e.g. “markets”) align with the rest "
        "of your topics. This lens is used for search, ranking, and summaries."
    )
    if prof.combined_interest_context:
        _markdown_block(prof.combined_interest_context)
    else:
        st.caption("No combined lens text for this run.")
    for interest in prof.interests:
        note = (prof.interest_context_by_interest or {}).get(interest, "").strip()
        if note:
            st.markdown(f"**{_safe_inline(interest)}** — {_safe_inline(note)}")
        else:
            st.markdown(f"**{_safe_inline(interest)}** — _No separate note; bundled with your other interests._")

    with st.expander("Candidate pool (ranked by primary relevance score)", expanded=False):
        ranked = result.ranked_candidates
        if not ranked:
            st.caption("No ranked candidates returned for this run.")
        else:
            rows: list[dict[str, str | float]] = []
            for ra in sorted(ranked, key=lambda x: x.relevance_score, reverse=True):
                rows.append(
                    {
                        "Title": humanize_single_line(ra.title, max_len=200),
                        "Snippet": humanize_single_line(ra.snippet, max_len=160),
                        "Published": (ra.published_date or "—")[:32],
                        "Score": round(float(ra.relevance_score), 1),
                        "Link": ra.url,
                    }
                )
            st.dataframe(
                rows,
                column_config={
                    "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                    "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                },
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Top Stories")
    if not result.summaries:
        reason_msg = ""
        retriever_msg = ""
        summarizer_msg = ""
        for step in result.agent_trace:
            if step.get("agent") == "NewsRetrieverAgent":
                retriever_msg = str(step.get("output", ""))
            if step.get("agent") == "ReadabilityFilter":
                reason_msg = str(step.get("output", ""))
            if step.get("agent") == "SummarizerAgent":
                summarizer_msg = str(step.get("output", ""))
        st.warning(
            "No story summaries were produced for this run. "
            "This can happen when full-text retrieval or LLM summarization fails."
        )
        if summarizer_msg:
            st.caption(f"Summarizer diagnostics: {summarizer_msg}")
        if retriever_msg:
            st.caption(f"Retriever diagnostics: {retriever_msg}")
        if reason_msg:
            st.caption(f"Diagnostics: {reason_msg}")
    for idx, summary in enumerate(result.summaries):
        with st.container(border=True):
            st.markdown(f"### {idx + 1}. {_safe_inline(humanize_single_line(summary.headline, max_len=280))}")
            cols = st.columns([3, 2])
            with cols[0]:
                _markdown_block(summary.one_sentence_summary)
                st.markdown("**Why it matters**")
                _markdown_block(summary.why_it_matters)
                tab_content, tab_interest = st.tabs(["Content Summary", "Interest Summary"])
                with tab_content:
                    _bullet_lines(summary.content_summary_points)
                with tab_interest:
                    _bullet_lines(summary.interest_summary_points)
                pub = (summary.published_date or "").strip()
                if pub:
                    st.markdown(f"**Published:** {_safe_inline(humanize_single_line(pub, max_len=120))}")
                st.markdown(f"**Source:** {_safe_inline(humanize_single_line(summary.source_name, max_len=120))}")
                st.markdown(f"[Read source]({summary.source_url})")
                st.caption("Uncertainty / limitations")
                _uncertainty_plain(summary.uncertainty_or_limitations)
            with cols[1]:
                if idx < len(result.image_paths):
                    st.image(result.image_paths[idx], use_container_width=True)
                    st.caption(
                        "AI-generated image (synthetic). Not a real photograph or evidence; used for visual context only."
                    )
                else:
                    st.info("No image generated for this story.")

    st.subheader("Broadcast Script")
    script_plain = humanize_display_text(result.broadcast_script.full_script)
    st.text_area("Anchor Script", value=script_plain, height=240)

    st.subheader("Audio Briefing")
    if result.audio_path:
        st.audio(result.audio_path)
    else:
        st.info("No narration yet. Click **Record Briefings** when script and visuals are ready.")

    st.subheader("Storyboard Video")
    if result.video_bytes:
        st.video(result.video_bytes)
        st.download_button(
            "Download Storyboard Video",
            data=result.video_bytes,
            file_name="ai_news_buddy_storyboard.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        st.caption("Storyboard style video from still images and narration (length follows audio).")
    else:
        st.info("No storyboard cut yet. Use **Record Briefings** after script and visuals are ready.")

    with st.expander("Agent Trace and Intermediate Outputs"):
        for step in result.agent_trace:
            agent = str(step.get("agent", ""))
            st.markdown(f"**{agent}**")
            output = step.get("output", "")
            if agent == "ModelRoutingReport" and isinstance(output, dict):
                used = output.get("used_models", {}) or {}
                st.markdown(
                    "\n".join(
                        [
                            f"- Text model: `{used.get('text') or 'N/A'}`",
                            f"- Image model: `{used.get('image') or 'N/A'}`",
                            f"- Voice model: `{used.get('voice') or 'N/A'}`",
                        ]
                    )
                )
                failures = output.get("failures", {}) or {}
                shown_any = False
                if isinstance(failures, dict):
                    for modality in ("text", "image", "voice"):
                        failures_for_modality = failures.get(modality) or []
                        if failures_for_modality:
                            shown_any = True
                            st.markdown(f"`{modality}` fallback failures: {len(failures_for_modality)}")
                if not shown_any:
                    st.caption("No model fallback failures recorded.")
            else:
                rendered = humanize_display_text(output).strip()
                st.code(rendered if rendered else str(output))


st.set_page_config(page_title="AI News Buddy", layout="wide")
st.title("AI News Buddy")
st.caption(
    "Personalized AI news broadcaster: retrieves stories, summarizes what matters, creates editorial visuals, and prepares a broadcast script."
)

if "briefing_result" not in st.session_state:
    st.session_state.briefing_result = None
if "pipeline_busy" not in st.session_state:
    st.session_state.pipeline_busy = False

with st.sidebar:
    st.header("Briefing Setup")
    interests_raw = st.text_input(
        "Interests (comma-separated)",
        "AI, robotics, markets",
        help=_HELP_INTERESTS,
    )
    region = st.text_input("Region", "US", help=_HELP_REGION)
    max_stories = st.slider(
        "Number of stories",
        min_value=1,
        max_value=6,
        value=3,
        help=_HELP_MAX_STORIES,
    )
    candidate_pool_size = st.slider(
        "Candidate pool size (recommended: 25)",
        min_value=5,
        max_value=50,
        value=25,
        step=1,
        help=_HELP_CANDIDATE_POOL,
    )
    st.caption("Use the buttons on the main page to run each pipeline step in order.")

config = load_config()
orchestrator = NewsBuddyOrchestrator(config)

_gen_title, _gen_help = st.columns([0.88, 0.12])
with _gen_title:
    st.subheader("Generate content")
with _gen_help:
    with st.popover("?"):
        st.markdown(_HELP_GENERATE_POPOVER.strip())
optimize_video_script = True

r_state = st.session_state.briefing_result
_news_done = bool(r_state and r_state.summaries)
_visuals_done = bool(r_state and r_state.image_paths)
_script_done = bool(r_state and (r_state.broadcast_script.full_script or "").strip())
_record_done = bool(r_state and r_state.audio_path and r_state.video_bytes)

_step_states = {
    1: "done" if _news_done else "ready",
    2: "locked" if not _news_done else ("done" if _visuals_done else "ready"),
    3: "locked" if not _news_done else ("done" if _script_done else "ready"),
    4: "locked" if not (_news_done and _visuals_done and _script_done) else ("done" if _record_done else "ready"),
}

_palette = {
    "ready": ("#16a34a", "#16a34a", "#ffffff"),  # green
    "done": ("#ef4444", "#ef4444", "#ffffff"),   # red
    "locked": ("#374151", "#4b5563", "#9ca3af"),  # gray
}

def _button_css(step: int, state: str) -> str:
    bg, border, fg = _palette[state]
    return (
        f".st-key-stepbtn_{step} button {{"
        f"background-color: {bg} !important;"
        f"border: 1px solid {border} !important;"
        f"color: {fg} !important;"
        "}"
    )

st.markdown(
    "<style>"
    + "".join(_button_css(i, _step_states[i]) for i in (1, 2, 3, 4))
    + ".stButton button:disabled { opacity: 1 !important; }"
    + ".stButton button:not(:disabled) { transition: transform 0.14s ease, box-shadow 0.14s ease, filter 0.14s ease; cursor: pointer !important; }"
    + ".stButton button:not(:disabled):hover { transform: translateY(-1px) scale(1.01); box-shadow: 0 0 0 1px rgba(255,255,255,0.22), 0 8px 18px rgba(0,0,0,0.35); filter: brightness(1.06); }"
    + ".stButton button:disabled { cursor: not-allowed !important; box-shadow: none !important; transform: none !important; filter: none !important; }"
    + "</style>",
    unsafe_allow_html=True,
)

r0 = st.session_state.briefing_result

with st.container(key="stepbtn_1"):
    clicked_find_news = st.button(
        "Find News",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state.pipeline_busy or _step_states[1] == "locked",
        help=_HELP_FIND_NEWS,
    )
if clicked_find_news:
    st.session_state.pipeline_busy = True
    try:
        with st.status("Finding and briefing stories…", expanded=True) as status:
            def _prog(stage: str, detail: str) -> None:
                status.update(label=f"{stage}…", state="running")
                status.write(f"- {detail}")

            res, _ = orchestrator.gather_summaries_only(
                interests_raw=interests_raw,
                region=region,
                max_stories=max_stories,
                candidate_pool_size=candidate_pool_size,
                persona=str(st.session_state.get("buddy_persona", "Lisa - professional anchor")),
                tone=str(st.session_state.get("buddy_tone", "professional")),
                max_video_seconds=0,
                use_live_search=True,
                progress_callback=_prog,
            )
            st.session_state.briefing_result = res
            status.update(label="Stories ready.", state="complete")
    finally:
        st.session_state.pipeline_busy = False
    st.success("News desk is set—stories and context are loaded.")
    st.rerun()

st.divider()

persona_col, tone_col = st.columns(2)
with persona_col:
    st.selectbox(
        "Persona",
        options=[
            "Lisa - professional anchor",
            "Friendly buddy",
            "Executive brief",
        ],
        key="buddy_persona",
        help=_HELP_PERSONA,
    )
with tone_col:
    st.selectbox(
        "Tone",
        options=["concise", "professional", "casual", "energetic"],
        index=1,
        key="buddy_tone",
        help=_HELP_TONE,
    )

row_mid_a, row_mid_b = st.columns(2)
with row_mid_a:
    with st.container(key="stepbtn_2"):
        clicked_draft_visuals = st.button(
            "Draft Visuals",
            type="secondary",
            use_container_width=True,
            disabled=st.session_state.pipeline_busy or _step_states[2] == "locked",
            help=_HELP_DRAFT_VISUALS,
        )
    if clicked_draft_visuals:
        st.session_state.pipeline_busy = True
        try:
            with st.status("Drafting visuals…", expanded=True) as status:
                def _prog2(stage: str, detail: str) -> None:
                    status.update(label=f"{stage}…", state="running")
                    status.write(f"- {detail}")

                if r0 is None:
                    st.error("Run **Find News** first.")
                    st.stop()
                styled = _profile_with_ui_style(r0.user_profile)
                vp, paths, tr = orchestrator.generate_story_images(
                    styled,
                    r0.summaries,
                    progress_callback=_prog2,
                )
                st.session_state.briefing_result = _merge_result(
                    r0,
                    user_profile=styled,
                    visual_prompts=vp,
                    image_paths=paths,
                    agent_trace=tr,
                )
                status.update(label="Visuals ready.", state="complete")
        finally:
            st.session_state.pipeline_busy = False
        st.success("Art department delivered—story images are ready.")
        st.rerun()

with row_mid_b:
    with st.container(key="stepbtn_3"):
        clicked_prepare_script = st.button(
            "Prepare Script",
            type="secondary",
            use_container_width=True,
            disabled=st.session_state.pipeline_busy or _step_states[3] == "locked",
            help=_HELP_PREPARE_SCRIPT,
        )
    if clicked_prepare_script:
        st.session_state.pipeline_busy = True
        try:
            with st.status("Preparing anchor script…", expanded=True) as status:
                def _prog3(stage: str, detail: str) -> None:
                    status.update(label=f"{stage}…", state="running")
                    status.write(f"- {detail}")

                if r0 is None:
                    st.error("Run **Find News** first.")
                    st.stop()
                styled = _profile_with_ui_style(r0.user_profile)
                script, tr = orchestrator.generate_anchor_script(
                    styled,
                    r0.summaries,
                    generate_video=optimize_video_script,
                    progress_callback=_prog3,
                )
                st.session_state.briefing_result = _merge_result(
                    r0,
                    user_profile=styled,
                    broadcast_script=script,
                    agent_trace=tr,
                )
                status.update(label="Script ready.", state="complete")
        finally:
            st.session_state.pipeline_busy = False
        st.success("Prompter copy is in—your anchor script is ready.")
        st.rerun()

with st.container(key="stepbtn_4"):
    clicked_record_briefings = st.button(
        "Record Briefings",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state.pipeline_busy or _step_states[4] == "locked",
        help=_HELP_RECORD_BRIEFINGS,
    )
if clicked_record_briefings:
    st.session_state.pipeline_busy = True
    try:
        with st.status("Recording narration and storyboard…", expanded=True) as status:
            def _prog_av(stage: str, detail: str) -> None:
                status.update(label=f"{stage}…", state="running")
                status.write(f"- {detail}")

            r1 = st.session_state.briefing_result
            if r1 is None:
                st.error("Run **Find News**, then **Prepare Script** and **Draft Visuals** before recording.")
                st.stop()
            styled = _profile_with_ui_style(r1.user_profile)
            audio_path, video_bytes, tr = orchestrator.generate_narration_and_storyboard_video(
                styled,
                r1.broadcast_script,
                r1.summaries,
                r1.image_paths,
                progress_callback=_prog_av,
            )
            st.session_state.briefing_result = _merge_result(
                r1,
                user_profile=styled,
                audio_path=audio_path,
                video_bytes=video_bytes,
                agent_trace=tr,
            )
            status.update(label="Recording complete.", state="complete")
    finally:
        st.session_state.pipeline_busy = False
    st.success("Control room wrapped—audio and video are ready.")
    st.rerun()

st.divider()
if st.session_state.briefing_result is None:
    st.info("Start with **Find News** to load interest analysis, candidates, and top stories.")
else:
    _render_briefing_result(st.session_state.briefing_result)

st.divider()
st.caption(
    "Safety: summaries are grounded to retrieved snippets and source URLs. Images are AI-generated and may be synthetic."
)
