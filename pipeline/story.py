"""OpenAI planner — kids stories or deep Principal-Architect system-design videos."""

from __future__ import annotations

import re
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from pipeline.config import (
    STORY_MODEL,
    STORY_TIMEOUT_SEC,
    SYSTEM_DESIGN_MAX_SCENES,
    SYSTEM_DESIGN_MAX_TOKENS,
    SYSTEM_DESIGN_MIN_PLAN_SEC,
    SYSTEM_DESIGN_MIN_SCENES,
    SYSTEM_DESIGN_MODEL,
    SYSTEM_DESIGN_TIMEOUT_SEC,
    ContentMode,
    Language,
    clamp_duration_sec,
    scenes_for_duration,
)

CameraMotion = Literal[
    "slow_zoom_in",
    "slow_zoom_out",
    "pan_left",
    "pan_right",
    "ken_burns",
    "focus_node",
]

PREFERRED_DIAGRAM_IDS = (
    "arch",
    "read_path",
    "write_path",
    "consistency_and_dlq",
    "scaling_bottlenecks",
)


class SystemDiagram(BaseModel):
    """Shared Mermaid diagram referenced by multiple scenes."""

    diagram_id: str = Field(
        description="Short id: arch, read_path, write_path, consistency_and_dlq, scaling_bottlenecks"
    )
    title: str
    kind: Literal["flowchart", "sequence"] = "flowchart"
    mermaid_code: str = Field(
        description="Valid Mermaid source only — no markdown fences, no HTML/CSS."
    )


class StoryScene(BaseModel):
    scene_number: int
    narration: str = Field(description="Spoken narration for this scene.")
    emotion: Literal["happy", "excited", "calm", "sad", "suspense", "outro"]
    image_prompt_en: str = Field(
        description=(
            "Kids: English prompt to CREATE one cartoon character GIF image "
            "(not a video). System design: visual highlight cue for editor."
        )
    )
    camera_motion: CameraMotion = "slow_zoom_in"
    gif_motion: str = Field(
        default="",
        description=(
            "Kids only: how the character GIF should move "
            "(e.g. 'bounce and wink', 'swim left-right with fin kick', 'happy dance'). "
            "Empty for system design."
        ),
    )
    # Kids story fields (unused for SD)
    diagram_boxes: list[str] = Field(default_factory=list)
    diagram_flows: list[str] = Field(default_factory=list)
    mermaid_code: str = Field(
        default="",
        description="Optional scene-specific Mermaid zoom; else inherit diagram_id.",
    )
    # System-design storyboard fields
    section_title: str = Field(
        default="",
        description='System design: e.g. "1. Traffic Math & Bottlenecks at 5M DAU"',
    )
    diagram_id: str = Field(
        default="",
        description="System design: which shared diagram to show.",
    )
    highlight_cue: str = Field(
        default="",
        description="Exact Mermaid node IDs or participant names to spotlight.",
    )
    example_callout: str = Field(
        default="",
        description="Concrete code/command/math on screen, e.g. SET NX PX or RPS formula.",
    )
    architect_tip: str = Field(
        default="",
        description="One-line pro recommendation or trade-off warning badge.",
    )
    on_screen_text: list[str] = Field(
        default_factory=list,
        description="System design: 3–4 short bullet chips.",
    )
    timestamp_hint: str = Field(
        default="",
        description="Realistic 25–35s window, e.g. 00:45–01:15.",
    )

    @field_validator("camera_motion", mode="before")
    @classmethod
    def normalize_camera(cls, value: object) -> str:
        text = str(value or "slow_zoom_in").strip().replace(" ", "_").lower()
        allowed = {
            "slow_zoom_in",
            "slow_zoom_out",
            "pan_left",
            "pan_right",
            "ken_burns",
            "focus_node",
        }
        return text if text in allowed else "slow_zoom_in"


class StoryPlan(BaseModel):
    title: str
    character_bible_en: str = Field(
        description=(
            "Kids: character lock. System design: visual style lock "
            "(dark #0F172A, cyan Redis, orange Kafka, green success)."
        )
    )
    mood: str
    music_prompt: str
    diagrams: list[SystemDiagram] = Field(
        default_factory=list,
        description="System design: 4–5 master Mermaid diagrams. Empty for kids stories.",
    )
    scenes: list[StoryScene]


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _strip_mermaid_fences(code: str) -> str:
    text = (code or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("mermaid"):
            text = text[7:].lstrip()
    if "<style" in text.lower() or "chatgpt-mermaid" in text.lower():
        keep = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "{" in s or "}" in s or s.startswith("@"):
                continue
            keep.append(line)
        text = "\n".join(keep).strip()
    return text


def _mermaid_kind_ok(code: str) -> bool:
    text = _strip_mermaid_fences(code)
    return bool(
        re.search(r"^\s*(flowchart|graph|sequenceDiagram)\b", text, re.I | re.M)
    )


def _topology_issues(code: str) -> list[str]:
    """Heuristic lint for inaccurate Redis/Kafka topology."""
    issues: list[str] = []
    raw = _strip_mermaid_fences(code)
    compact = re.sub(r"\s+", " ", raw)
    # Redis talking to DB without API in the middle is a common LLM mistake.
    if re.search(
        r"(?i)Redis[^;\n]{0,40}(-->|->>|-->>).{0,40}(Database|Postgres|PostgreSQL|Primary DB|DB\b)",
        compact,
    ):
        if not re.search(r"(?i)(API|Server|Gateway|Service).{0,30}(-->|->>)", compact):
            issues.append(
                "Redis must not query DB directly — API handles cache miss → DB"
            )
    # Kafka → DB without consumer/worker
    if re.search(
        r"(?i)Kafka[^;\n]{0,40}(-->|->>|-->>).{0,40}(Database|Postgres|PostgreSQL|Primary DB|DB\b)",
        compact,
    ):
        if not re.search(r"(?i)(Consumer|Worker|ConsumerGroup)", raw):
            issues.append(
                "Kafka must not write DB directly — put a Consumer/Worker between them"
            )
    return issues


def validate_system_design_plan(plan: StoryPlan) -> list[str]:
    """Return human-readable validation errors (empty = OK)."""
    errors: list[str] = []
    if len(plan.scenes) < 8:
        errors.append(f"Need at least 8 scenes (got {len(plan.scenes)}; prefer 10–14)")
    if len(plan.diagrams) < 3:
        errors.append(f"Need at least 3 diagrams (got {len(plan.diagrams)}; prefer 4–5)")

    diagram_ids = {}
    for d in plan.diagrams:
        did = (d.diagram_id or "").strip().replace(" ", "_").lower()
        code = _strip_mermaid_fences(d.mermaid_code)
        if not did:
            errors.append("Diagram missing diagram_id")
            continue
        diagram_ids[did] = code
        if not _mermaid_kind_ok(code):
            errors.append(f"Diagram '{did}' must start with flowchart or sequenceDiagram")
        for tip in _topology_issues(code):
            errors.append(f"Diagram '{did}': {tip}")

    short_scenes = 0
    for sc in plan.scenes:
        wc = _word_count(sc.narration)
        if wc < 45:
            short_scenes += 1
        did = (sc.diagram_id or "").strip().replace(" ", "_").lower()
        if did and did not in diagram_ids:
            errors.append(f"Scene {sc.scene_number} references unknown diagram_id '{did}'")
        if sc.mermaid_code and not _mermaid_kind_ok(sc.mermaid_code):
            errors.append(
                f"Scene {sc.scene_number} mermaid_code must start with flowchart/sequenceDiagram"
            )
    if short_scenes:
        errors.append(
            f"{short_scenes} scene(s) have narration under 45 words (need 60–110)"
        )
    return errors


def _fallback_arch_mermaid() -> str:
    return (
        "flowchart LR\n"
        "  subgraph Clients\n"
        "    U[Users]\n"
        "  end\n"
        "  subgraph Edge\n"
        "    CDN[CDN]\n"
        "    LB[L7 Load Balancer]\n"
        "    GW[API Gateway]\n"
        "  end\n"
        "  subgraph Compute\n"
        "    API[Stateless API]\n"
        "  end\n"
        "  subgraph RedisHot\n"
        "    R[(Redis Cluster)]\n"
        "  end\n"
        "  subgraph KafkaAsync\n"
        "    K[[Kafka]]\n"
        "    C[Consumer Workers]\n"
        "    DLQ[[DLQ]]\n"
        "  end\n"
        "  subgraph Storage\n"
        "    DB[(PostgreSQL)]\n"
        "  end\n"
        "  U --> CDN --> LB --> GW --> API\n"
        "  API --> R\n"
        "  API -->|cache miss| DB\n"
        "  API --> K --> C --> DB\n"
        "  C -.-> DLQ\n"
    )


def _normalize_sd_plan(plan: StoryPlan) -> StoryPlan:
    plan.character_bible_en = (
        "dark #0F172A engineering diagrams; cyan Redis; orange Kafka; "
        "green success; purple DB; Mermaid only; no people"
    )
    cleaned: list[SystemDiagram] = []
    for d in plan.diagrams or []:
        d.mermaid_code = _strip_mermaid_fences(d.mermaid_code)
        d.diagram_id = (d.diagram_id or "arch").strip().replace(" ", "_").lower()
        if d.mermaid_code:
            cleaned.append(d)
    if not cleaned:
        cleaned = [
            SystemDiagram(
                diagram_id="arch",
                title="System Architecture",
                kind="flowchart",
                mermaid_code=_fallback_arch_mermaid(),
            )
        ]
    plan.diagrams = cleaned
    ids = {d.diagram_id for d in plan.diagrams}
    default_id = plan.diagrams[0].diagram_id
    for sc in plan.scenes:
        if sc.mermaid_code:
            sc.mermaid_code = _strip_mermaid_fences(sc.mermaid_code)
        sc.diagram_boxes = []
        sc.diagram_flows = []
        did = (sc.diagram_id or "").strip().replace(" ", "_").lower()
        sc.diagram_id = did if did in ids else default_id
        if not sc.on_screen_text and sc.highlight_cue:
            sc.on_screen_text = [sc.highlight_cue[:48]]
        if not sc.image_prompt_en:
            sc.image_prompt_en = sc.highlight_cue or sc.section_title or "architecture"
        if len(sc.on_screen_text) > 4:
            sc.on_screen_text = sc.on_screen_text[:4]
    return plan


def _sd_system_prompt(target_sec: int, min_scenes: int, max_scenes: int) -> str:
    minutes_approx = target_sec / 60.0
    return (
        "You are a Principal System Architect producing an in-depth educational video "
        "for software engineers (ByteByteGo / interview-prep depth). "
        "NO characters, people, mascots, or DALL·E/GPT Image prompts — Mermaid diagrams only.\n\n"
        "NARRATION LANGUAGE: Natural conversational Hinglish (Hindi + English mixed). "
        "Keep standard English technical terms intact — never awkwardly translate: "
        "Cache Stampede, Throughput, Partition, Idempotency, p99 Latency, Consumer Group, "
        "DLQ, Read-Your-Own-Writes, SET NX PX, HTTP 202 Accepted.\n\n"
        "DEPTH RULE — for EVERY component or pattern, narration MUST cover:\n"
        "1) Scale & Numbers — concrete traffic math (e.g. 5M DAU ≈ 6k–15k peak RPS, <50ms p99).\n"
        "2) Problem / Failure — what breaks at scale without it "
        "(pool exhaustion, stampede, lock contention).\n"
        "3) How it works — step-by-step with real commands/payloads "
        "(SET lock:user:123 uuid NX PX 5000, Kafka partition by user_id, PgBouncer).\n"
        "4) Architect tip & trade-offs — TTL jitter 300s±30s, DLQ, idempotent consumers, etc.\n\n"
        "PACING:\n"
        f"- Target runtime ≈ {target_sec}s (~{minutes_approx:.1f} min).\n"
        f"- Create {min_scenes}–{max_scenes} detailed scenes (never fewer than 8).\n"
        "- Each scene narration MUST be 60–110 words (~25–35 seconds spoken).\n"
        "- Never give high-level one-liners.\n\n"
        "DIAGRAMS (diagrams array — 4 to 5 rich Mermaid sources):\n"
        "- Required ids when topic fits: arch, read_path, write_path, "
        "consistency_and_dlq, scaling_bottlenecks.\n"
        "- Use flowchart LR or flowchart TD with styled subgraph blocks; "
        "sequenceDiagram with alt/opt/par and Note over callouts.\n"
        "- TOPOLOGY LAW: Redis NEVER queries the database — Stateless API handles "
        "cache miss and queries DB. Consumers sit between Kafka and Database — "
        "never draw Kafka --> Primary Database without a Consumer/Worker node.\n"
        "- Mermaid ONLY inside mermaid_code — zero markdown fences, zero HTML/CSS.\n"
        "- classDef only on flowcharts (never on sequenceDiagram).\n\n"
        "EACH SCENE MUST INCLUDE:\n"
        "- section_title (e.g. '1. Traffic Math & Bottlenecks at 5M DAU')\n"
        "- diagram_id mapping to diagrams[]\n"
        "- highlight_cue = exact Mermaid node IDs or participant names\n"
        "- example_callout = concrete command/math/HTTP status for on-screen display\n"
        "- architect_tip = one-line pro recommendation or trade-off warning\n"
        "- on_screen_text = 3–4 short bullets\n"
        "- camera_motion one of: slow_zoom_in, slow_zoom_out, pan_left, pan_right, "
        "ken_burns, focus_node\n"
        "- timestamp_hint with realistic 25–35s windows\n"
        "- optional scene mermaid_code only if a zoomed variant helps; else leave empty\n\n"
        "character_bible_en = dark #0F172A, cyan Redis, orange Kafka, green success, purple DB.\n"
        "music_prompt = calm dark tech/ambient instrumental (no vocals).\n"
        "FINAL scene emotion='outro' with architecture recap + thank-you (still 60–110 words)."
    )


def _ensure_gif_image_prompt(text: str) -> str:
    """Kids path: force Disney GIF wording + quality tags (not video)."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        cleaned = (
            "Create an animated Disney Pixar cartoon GIF of the same hero mid-action "
            "in a lush colorful world with golden-hour light and deep background scenery"
        )
    low = cleaned.lower()
    for bad in ("video", "film clip", "movie shot", "cinematic sequence", "mp4"):
        if bad in low:
            cleaned = re.sub(re.escape(bad), "Disney cartoon GIF", cleaned, flags=re.I)
            low = cleaned.lower()
    if "gif" not in low:
        cleaned = f"Create an animated Disney Pixar cartoon GIF of {cleaned}"
    elif not low.startswith("create"):
        cleaned = f"Create an animated Disney Pixar cartoon GIF: {cleaned}"
    quality_bits = []
    if "golden" not in cleaned.lower() and "lighting" not in cleaned.lower():
        quality_bits.append("warm golden-hour Disney lighting")
    if "background" not in cleaned.lower() and "landscape" not in cleaned.lower():
        quality_bits.append("rich layered background depth")
    if quality_bits:
        cleaned = f"{cleaned}, {', '.join(quality_bits)}"
    return cleaned[:700]


def _call_parse(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int,
) -> StoryPlan:
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=StoryPlan,
        max_completion_tokens=max_tokens,
        timeout=timeout,
    )
    plan = completion.choices[0].message.parsed
    if plan is None:
        raise RuntimeError("OpenAI returned an empty story plan.")
    return plan


def _expand_short_narrations(client: OpenAI, plan: StoryPlan, model: str) -> StoryPlan:
    """Second-pass expand narrations that are still too short."""
    for sc in plan.scenes:
        if _word_count(sc.narration) >= 45:
            continue
        prompt = (
            "Expand this system-design video narration to 60–110 words of Hinglish "
            "(keep English tech terms). Cover scale numbers, failure scenario, "
            "how it works with a concrete example, and an architect tip.\n\n"
            f"Section: {sc.section_title or sc.scene_number}\n"
            f"Highlight: {sc.highlight_cue}\n"
            f"Callout: {sc.example_callout}\n"
            f"Current narration:\n{sc.narration}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You expand technical video narration only. Return plain text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=400,
                timeout=60,
            )
            text = (resp.choices[0].message.content or "").strip()
            if _word_count(text) >= 45:
                sc.narration = text
        except Exception:
            continue
    return plan


def generate_story(
    client: OpenAI,
    user_topic: str,
    language: Language,
    *,
    duration_sec: int = 120,
    minutes: int | None = None,
    title_hint: str | None = None,
    max_scenes_override: int | None = None,
    content_mode: ContentMode = "story",
) -> StoryPlan:
    if minutes is not None and duration_sec == 120:
        duration_sec = int(minutes) * 60
    target_sec = clamp_duration_sec(duration_sec)
    min_scenes, max_scenes = scenes_for_duration(target_sec)
    if max_scenes_override is not None:
        max_scenes = max(1, int(max_scenes_override))
        # Kids GIF stories: aim near override so ~8–10s beats fill the runtime.
        min_scenes = max(min_scenes, max(1, max_scenes - 2))
        min_scenes = min(min_scenes, max_scenes)

    content_mode = "system_design" if content_mode == "system_design" else "story"
    title_line = f" Preferred title hint: {title_hint}." if title_hint else ""

    if content_mode == "system_design":
        # Prefer depth over shallow short UI durations.
        target_sec = max(target_sec, SYSTEM_DESIGN_MIN_PLAN_SEC)
        target_sec = min(target_sec, 300)
        min_scenes = SYSTEM_DESIGN_MIN_SCENES
        max_scenes = SYSTEM_DESIGN_MAX_SCENES
        if max_scenes_override is not None:
            max_scenes = max(8, int(max_scenes_override))
            min_scenes = max(8, min(min_scenes, max_scenes))

        model = SYSTEM_DESIGN_MODEL
        timeout = SYSTEM_DESIGN_TIMEOUT_SEC
        max_tokens = SYSTEM_DESIGN_MAX_TOKENS
        system_prompt = _sd_system_prompt(target_sec, min_scenes, max_scenes)
        user_prompt = (
            f"Produce a FULL {target_sec}-second Principal Architect system-design "
            f"explainer video package about: {user_topic}.{title_line}\n"
            f"Deliver {min_scenes}–{max_scenes} deep Hinglish scenes (60–110 words each) "
            "AND 4–5 accurate Mermaid diagrams (arch, read_path, write_path, "
            "consistency_and_dlq, scaling_bottlenecks when relevant). "
            "Enforce correct topology: API owns cache-miss→DB; Consumer between Kafka and DB."
        )

        plan = _call_parse(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        plan = _normalize_sd_plan(plan)
        errors = validate_system_design_plan(plan)
        if errors:
            repair = (
                "Your previous storyboard FAILED validation. Regenerate the FULL package "
                "fixing ALL issues. Do not shorten narrations.\nIssues:\n- "
                + "\n- ".join(errors)
                + f"\n\nOriginal topic: {user_topic}"
            )
            plan2 = _call_parse(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=repair,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            plan2 = _normalize_sd_plan(plan2)
            errors2 = validate_system_design_plan(plan2)
            # Prefer repaired plan if better; else expand short narrations on best.
            if not errors2 or len(errors2) < len(errors):
                plan = plan2
                errors = errors2
            if errors:
                plan = _expand_short_narrations(client, plan, model)
                plan = _normalize_sd_plan(plan)
                errors = validate_system_design_plan(plan)
            if len(plan.scenes) < 8:
                raise RuntimeError(
                    "System-design storyboard too shallow after retry: "
                    + "; ".join(errors[:6])
                )
        return plan

    # --- Kids cartoon rhymes / musical GIF storyboard ---
    w_total = max(4, int(target_sec * 2.4))
    per = max(3, w_total // max(1, (min_scenes + max_scenes) // 2))
    w_lo, w_hi = max(3, int(per * 0.75)), max(5, int(per * 1.25))
    minutes_approx = target_sec / 60.0
    if language == "hindi":
        lang_rules = (
            "Narration: natural kids Hindi (Devanagari), sing-song rhyme-friendly lines; "
            "character names may stay English."
        )
        outro = "FINAL scene emotion='outro' with a short rhyming thank-you."
    else:
        lang_rules = (
            "Narration: clear kids English with gentle rhyme / chorus energy "
            "(British or Indian English as fits the topic)."
        )
        outro = "FINAL scene emotion='outro' with a short rhyming thank-you."

    system_prompt = (
        "You write kids Pixar/Disney musical cartoon RHYME storyboards for this pipeline:\n"
        "  (1) FIRST we generate a locked CHARACTER REFERENCE from character_bible_en.\n"
        "  (2) THEN each scene becomes an 8–10 second Disney GIF story beat "
        "(OpenAI still + motion) — NOT a video file from the model.\n"
        "  (3) We MERGE those GIFs into one video (music/VO optional).\n"
        "Never ask for video/mp4/film. Only GIF images + motion text.\n\n"
        "DISNEY GIF QUALITY BAR (match this look every time):\n"
        "Think ChatGPT Disney GIFs — e.g. a bright blue bird flying from snow-capped "
        "mountains across a misty village valley toward a terracotta rooftop with vines, "
        "lantern, golden-hour light, fluffy peach clouds, rich painterly Pixar/Tangled depth. "
        "Every image_prompt_en must demand that same polish: saturated color, layered "
        "foreground/midground/background, warm cinematic light, hero mid-action, "
        "NOT a blank studio selfie.\n\n"
        f"{lang_rules}\n\n"
        f"1. Spoken runtime ≈ {target_sec}s (~{minutes_approx:.1f} min); "
        f"{min_scenes}–{max_scenes} scenes; ~{w_lo}-{w_hi} words narration each.\n"
        "2. character_bible_en = locked English look for the cast (if topic is a duo like "
        "cat+mouse, lock BOTH: species, colors, size contrast). Never replace a mouse with "
        "extra cats. Same cast in every scene.\n"
        "3. image_prompt_en (English, concrete, 25–45 words) = CREATE the Disney GIF image:\n"
        "   - Lead with WHO + WHERE + ACTION (e.g. 'grey cat and brown mouse splash in "
        "a sunny swimming pool…'). Place matters — pool/water must be visible if story is "
        "about a pool.\n"
        "   - Disney Pixar 3D, golden light; no on-image text.\n"
        "   - Never say video/film/movie/mp4.\n"
        "   - Never invent a crowd of clone cats; keep the exact cast from character_bible.\n"
        "4. gif_motion (English, REQUIRED) = motion recipe matching the beat, e.g.\n"
        "   'fly left-to-right with wing flap and soft parallax pan', "
        "'gentle bounce and wink', 'swim side to side with fin kick', "
        "'happy dance with arm sway'.\n"
        "5. Leave diagrams=[], diagram_id, highlight_cue, on_screen_text, "
        "section_title, example_callout, architect_tip empty.\n"
        "6. music_prompt = soft instrumental kids bed only (no vocals) for later sync.\n"
        "7. mood = cheerful nursery-rhyme feel.\n"
        f"8. {outro}"
    )
    user_prompt = (
        f"Make a {target_sec}s {language} kids musical cartoon RHYME about: {user_topic}."
        f"{title_line}\n"
        "For each scene return: singable narration, a RICH Disney-quality image_prompt_en "
        "('Create an animated Disney Pixar cartoon GIF of…' with environment + lighting), "
        "and gif_motion. Quality must match polished Disney GIFs (bird over mountains to "
        "rooftop style). We merge GIFs into a video and sync music."
    )
    plan = _call_parse(
        client,
        model=STORY_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=STORY_TIMEOUT_SEC,
        max_tokens=8192,
    )
    plan.diagrams = []
    for sc in plan.scenes:
        sc.image_prompt_en = _ensure_gif_image_prompt(sc.image_prompt_en)
        if not (sc.gif_motion or "").strip():
            sc.gif_motion = "gentle bounce, soft blink, happy face sway"
        sc.gif_motion = " ".join(sc.gif_motion.split())[:180]
    return plan
