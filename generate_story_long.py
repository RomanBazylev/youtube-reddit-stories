"""
Long-form Reddit Stories video: compilation of 3-5 stories (8-12 min).
Pipeline: LLM generates multi-story script → edge-tts → Pexels clips → ffmpeg → upload
"""

import asyncio
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import edge_tts
import requests

# ── Constants ──────────────────────────────────────────────────────────
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_PATH = BUILD_DIR / "voiceover.mp3"
MUSIC_PATH = BUILD_DIR / "music.mp3"
OUTPUT_PATH = BUILD_DIR / "output_story_long.mp4"
TITLE_HISTORY_PATH = Path("title_history.json")
MAX_TITLE_HISTORY = 40

TARGET_W, TARGET_H = 1280, 720
FPS = 30
FFMPEG_PRESET = "medium"
FFMPEG_CRF = "23"

TTS_VOICES = [
    "en-US-AndrewMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-GuyNeural",
]
TTS_RATE_OPTIONS = ["+0%", "+3%", "+5%"]

TTS_PRONUNCIATION_FIXES = {
    "AITA": "am I the A-hole",
    "TIFU": "today I effed up",
    "OP": "O-P",
    "TL;DR": "T-L-D-R",
    "TLDR": "T-L-D-R",
    "MIL": "mother in law",
    "FIL": "father in law",
    "SIL": "sister in law",
    "BIL": "brother in law",
    "SO": "significant other",
    "GF": "girlfriend",
    "BF": "boyfriend",
    "NTA": "not the A-hole",
    "YTA": "you're the A-hole",
    "ESH": "everyone sucks here",
    "subreddit": "sub-reddit",
    "r/": "the subreddit ",
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

COMPILATION_THEMES = [
    "revenge stories where the bully gets what they deserve",
    "neighbor conflicts that escalated beyond belief",
    "family secrets revealed at the worst possible time",
    "wedding disasters with unexpected happy endings",
    "workplace drama where the boss got exposed",
    "entitled people getting instant karma",
    "inheritance fights that tore families apart",
    "roommate nightmares from absolute hell",
    "caught cheating stories with satisfying endings",
    "road trip disasters that became legendary",
    "school reunion stories with shocking twists",
    "online dating horror stories with wild endings",
    "malicious compliance that backfired perfectly",
    "betrayal stories where trust was broken forever",
    "stranger encounters that changed someone's life",
    "moving out stories with terrible landlords",
    "secrets overheard that should have stayed hidden",
    "small town scandals everyone still talks about",
    "holiday dinner fights that went nuclear",
    "parking lot confrontations with insane twists",
]

STORY_CHARACTERS = [
    "a retired teacher", "a college freshman", "a single parent",
    "a night shift nurse", "a delivery driver", "a small business owner",
    "a military veteran", "a young couple", "an elderly neighbor",
    "a high school senior", "a foster parent", "a bartender",
    "a new employee", "a wedding planner", "a park ranger",
]

PEXELS_QUERIES = [
    "person thinking alone", "city street night", "dramatic sunset",
    "empty room moody", "rain window reflection", "dark hallway",
    "person walking alone", "office cubicle work", "car driving night",
    "house suburban street", "courtroom justice", "argument conflict",
    "moving boxes house", "restaurant dinner table", "hospital corridor",
    "school hallway empty", "apartment door knockingr", "phone screen dark",
    "park bench alone", "kitchen home cooking", "parking lot night",
    "wedding venue empty", "road trip highway", "bar counter drinks",
    "mirror reflection person", "stairs climbing dark", "window looking out",
]

MUSIC_URLS = [
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Its_time_for_adventure/Komiku_-_05_-_Friends.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Daydream/Podington_Bear_-_Daydream.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
]

_DESCRIPTION_FOOTER = (
    "\n\n---\n"
    "Like & Subscribe for more wild Reddit stories every week! 🔔\n"
    "Drop your craziest story in the comments 👇\n\n"
    "#reddit #storytime #compilation #redditstories"
)

_CORE_TAGS = [
    "reddit", "storytime", "redditstories", "compilation",
    "askreddit", "aita", "truestories", "viral",
]

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"


# ── Helpers ────────────────────────────────────────────────────────────
def _clean_build_dir():
    if BUILD_DIR.is_dir():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg(cmd: list):
    print(f"[CMD] {' '.join(cmd[:8])}... ({len(cmd)} args)")
    subprocess.run(cmd, check=True)


def _probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    ).strip()
    return float(out)


def _fix_pronunciation(text: str) -> str:
    result = text
    for word, fix in TTS_PRONUNCIATION_FIXES.items():
        result = re.sub(re.escape(word), fix, result, flags=re.IGNORECASE)
    return result


def _groq_call(messages: list, temperature: float = 0.85, max_tokens: int = 8192, json_mode: bool = False) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(1, 3):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"[WARN] Groq attempt {attempt}: {exc}")
            time.sleep(5)
    return None


def _load_title_history() -> list[str]:
    if TITLE_HISTORY_PATH.is_file():
        try:
            return json.loads(TITLE_HISTORY_PATH.read_text("utf-8"))
        except Exception:
            pass
    return []


def _save_title_history(titles: list[str]):
    TITLE_HISTORY_PATH.write_text(
        json.dumps(titles[-MAX_TITLE_HISTORY:], ensure_ascii=False), encoding="utf-8"
    )


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Robustly parse JSON from LLM output, handling common quirks."""
    import ast

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # Extract the outermost { ... } to drop any preamble / postamble text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]

    # Attempt 1: direct json.loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix control characters inside string values
    cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: f'\\u{ord(m.group()):04x}', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: ast.literal_eval (handles single-quoted Python dicts)
    try:
        result = ast.literal_eval(cleaned)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Attempt 4: fix single quotes → double quotes for keys/values
    fixed = cleaned.replace("'", '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 5: fix unquoted keys  {title: "value"} → {"title": "value"}
    fixed2 = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    print(f"[WARN] All JSON parse attempts failed. First 300 chars: {raw[:300]}")
    return None


# ── Script Generation ────────────────────────────────────────────────
def generate_compilation_script() -> Optional[dict]:
    theme = random.choice(COMPILATION_THEMES)
    chars = random.sample(STORY_CHARACTERS, min(5, len(STORY_CHARACTERS)))
    story_count = random.choice([3, 4, 5])

    messages = [
        {"role": "system", "content": (
            "You are a top-tier Reddit storyteller narrating a YouTube compilation video. "
            "Your style: dramatic, engaging, conversational — like telling stories around a campfire.\n\n"
            "RULES:\n"
            "- Each sentence: max 15 words (for TTS narration).\n"
            "- Include specific details: names (fake), ages, locations, dollar amounts.\n"
            "- Use vivid dialogue: 'She looked at me and said: You have no idea what you just did.'\n"
            "- NO generic filler: 'you won't believe', 'this is crazy', 'trust me'.\n"
            "- Write in first person or narrator perspective.\n"
            "- Respond with ONLY a JSON object. No markdown, no commentary.\n"
        )},
        {"role": "user", "content": f"""Write a YouTube compilation video script: "{story_count} Insane Reddit Stories About {theme.title()}"

This is an 8-12 minute video. The script MUST be 1500-2000 words long.

CRITICAL: The "script" field must contain AT LEAST 1500 words. This is a LONG video, not a short.
If the script is under 1000 words, the video CANNOT be produced and will be rejected.

STRUCTURE (write ALL of these sections in full, do NOT skip any):
1. INTRO (50-70 words): Hook the viewer. Tease what's coming. Ask them to subscribe.
2. STORIES ({story_count} stories, each 300-400 words — this is CRITICAL):
   Each story MUST follow this exact structure:
   - TRANSITION (1-2 sentences): "Story number X takes us to..." or "But wait, story number two is even wilder..."
   - HOOK (2-3 sentences, ~40 words): Set up who the character is. Example: "Marcus, 34, was a quiet accountant in Portland. He had never been in trouble in his life. But one Thursday afternoon, everything changed."
   - SETUP (4-5 sentences, ~70 words): Describe the situation in detail. Include the subreddit, throwaway account name. What was the normal routine?
   - ESCALATION (5-7 sentences, ~100 words): Things go wrong. Include dialogue between characters. Direct quotes. Tension builds.
   - TWIST (2-3 sentences, ~50 words): The unexpected turn. The moment everything flips.
   - RESOLUTION (3-4 sentences, ~60 words): What happened after. The aftermath. Upvote count, community reaction.
   Characters to use: {', '.join(chars[:story_count])}
3. OUTRO (50-70 words): Recap the wildest moments. Ask which story was craziest. Subscribe CTA.

WORD COUNT MATH: Introduction (~60 words) + {story_count} stories x 350 words (~{story_count * 350} words) + Outro (~60 words) = ~{story_count * 350 + 120} words total.
You MUST hit at least 1500 words. Count carefully.

Return a JSON object with these exact keys:
- "title": string, engaging title max 90 chars with emoji
- "description": string, 5-8 lines with hashtags
- "tags": array of 15-20 strings
- "pexels_queries": array of 6-8 English search queries for footage
- "script": ONE STRING with the full narration (1500-2000 words), sentences separated by newlines. NOT an array."""},
    ]

    content = _groq_call(messages, temperature=0.9, max_tokens=16384, json_mode=True)
    if not content:
        return None
    data = _parse_llm_json(content)
    if not data:
        return None
    script = data.get("script", "")
    if isinstance(script, list):
        script = "\n".join(str(s) for s in script)
        data["script"] = script
    wc = len(script.split())
    print(f"[SCRIPT] {wc} words, {story_count} stories, theme: {theme}")
    if wc < 300:
        print("[WARN] Script too short (< 300 words), skipping")
        return None
    if wc < 800:
        print(f"[WARN] Script shorter than ideal ({wc} words), but usable")
    return data


# ── TTS ───────────────────────────────────────────────────────────────
async def _generate_tts(text: str, output_path: Path) -> list[dict]:
    voice = random.choice(TTS_VOICES)
    rate = random.choice(TTS_RATE_OPTIONS)
    tts_text = _fix_pronunciation(text)
    comm = edge_tts.Communicate(tts_text, voice, rate=rate)
    word_events = []
    with open(output_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                word_events.append({
                    "text": chunk["text"],
                    "offset": chunk["offset"] / 10_000_000,
                    "duration": chunk["duration"] / 10_000_000,
                })
    print(f"[TTS] {voice} rate={rate}, {len(word_events)} words")
    return word_events


def generate_tts(text: str) -> tuple[Path, list[dict]]:
    return AUDIO_PATH, asyncio.run(_generate_tts(text, AUDIO_PATH))


# ── Clips ─────────────────────────────────────────────────────────────
def _download_file(url: str, dest: Path):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_clips(extra_queries: list[str] = None, target: int = 35) -> list[Path]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    queries = list(extra_queries or [])
    base = [q for q in PEXELS_QUERIES if q not in queries]
    random.shuffle(base)
    queries.extend(base)

    headers = {"Authorization": api_key}
    paths, seen = [], set()
    idx = 0

    for query in queries:
        if len(paths) >= target:
            break
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params={"query": query, "per_page": 3, "orientation": "landscape"},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Pexels '{query}': {exc}")
            continue

        for video in resp.json().get("videos", []):
            vid_id = video.get("id")
            if vid_id in seen:
                continue
            seen.add(vid_id)
            hd = [f for f in video.get("video_files", []) if (f.get("height") or 0) >= 720]
            if not hd:
                continue
            best = min(hd, key=lambda f: abs((f.get("height") or 0) - 720))
            idx += 1
            clip_path = CLIPS_DIR / f"clip_{idx:03d}.mp4"
            try:
                _download_file(best["link"], clip_path)
                paths.append(clip_path)
            except Exception:
                pass
            if len(paths) >= target:
                break

    print(f"[CLIPS] Downloaded {len(paths)}")
    return paths


def download_music() -> Optional[Path]:
    for url in random.sample(MUSIC_URLS, len(MUSIC_URLS)):
        try:
            _download_file(url, MUSIC_PATH)
            return MUSIC_PATH
        except Exception:
            continue
    return None


# ── FFmpeg Assembly ──────────────────────────────────────────────────
def _prepare_clip(src: Path, dst: Path, duration: int = 5):
    vf = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},fps={FPS}"
    )
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src), "-t", str(duration),
        "-vf", vf, "-an", "-c:v", "libx264",
        "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, str(dst),
    ])


def _fmt_ass_time(s: float) -> str:
    cs = max(0, int(round(s * 100)))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def _safe_text(raw: str) -> str:
    t = raw.replace("\\", " ").replace("\n", " ")
    t = t.replace(":", " ").replace(";", " ").replace("'", "").replace('"', "")
    t = re.sub(r"\s+", " ", t).strip()
    return t or " "


def _group_words(events: list[dict], max_per: int = 5) -> list[dict]:
    if not events:
        return []
    lines, buf, start, end, kara = [], [], 0.0, 0.0, []
    for ev in events:
        s, d = ev["offset"], ev["duration"]
        if buf and (len(buf) >= max_per or (s - end) > 0.6):
            lines.append({"start": start, "end": end, "text": " ".join(buf), "words": list(kara)})
            buf, kara = [], []
        if not buf:
            start = s
        buf.append(ev["text"])
        kara.append({"text": ev["text"], "offset": s, "duration": d})
        end = s + d
    if buf:
        lines.append({"start": start, "end": end, "text": " ".join(buf), "words": list(kara)})
    return lines


def _write_ass(word_events: list[dict], ass_path: Path) -> Path:
    primary = "&H0000D4FF"
    secondary = "&H00FFFFFF"
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n"
        f"PlayResX: {TARGET_W}\nPlayResY: {TARGET_H}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Kara,DejaVu Sans,44,{primary},{secondary},&H00000000,&H80000000,"
        "1,0,0,0,100,100,1,0,1,3,2,2,30,30,80,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = _group_words(word_events)
    events = []
    for line in lines:
        s, e = line["start"], line["end"] + 0.15
        parts = []
        for w in line["words"]:
            dc = max(5, int(w["duration"] * 100))
            parts.append(f"{{\\kf{dc}}}{_safe_text(w['text']).upper()}")
        events.append(f"Dialogue: 0,{_fmt_ass_time(s)},{_fmt_ass_time(e)},Kara,,0,0,0,,{' '.join(parts)}")
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[SUBS] {len(events)} lines → {ass_path}")
    return ass_path


def assemble_video(clips: list[Path], voiceover: Path, word_events: list[dict], music: Optional[Path]) -> Path:
    temp = BUILD_DIR / "temp"
    temp.mkdir(exist_ok=True)

    prepared = []
    for i, clip in enumerate(clips):
        dst = temp / f"prep_{i:03d}.mp4"
        _prepare_clip(clip, dst, duration=5)
        prepared.append(dst)

    concat_file = temp / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in prepared), encoding="utf-8")
    silent = temp / "silent.mp4"
    _run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(silent)])

    voice_dur = _probe_duration(voiceover)
    clip_dur = _probe_duration(silent)
    final_dur = voice_dur + 1.5

    if clip_dur < voice_dur:
        looped = temp / "looped.mp4"
        _run_ffmpeg(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(silent),
                     "-t", f"{final_dur:.2f}", "-c", "copy", str(looped)])
        silent = looped

    ass_path = _write_ass(word_events, temp / "captions.ass")
    graded = temp / "graded.mp4"
    ass_esc = ass_path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("[", "\\[").replace("]", "\\]")
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(silent), "-vf", f"subtitles={ass_esc}",
                 "-t", f"{final_dur:.2f}", "-c:v", "libx264", "-preset", FFMPEG_PRESET,
                 "-crf", FFMPEG_CRF, "-an", str(graded)])

    voice_pad = f"apad=whole_dur={final_dur:.2f}"
    cmd = ["ffmpeg", "-y", "-i", str(graded), "-i", str(voiceover)]
    if music and music.exists():
        cmd.extend(["-stream_loop", "-1", "-i", str(music)])
        cmd.extend(["-filter_complex",
                     (f"[1:a]acompressor=threshold=-18dB:ratio=2.5:attack=5:release=120,{voice_pad}[va];"
                      "[va]asplit=2[va1][va2];"
                      "[2:a]highpass=f=80,lowpass=f=14000,volume=0.14[ma];"
                      "[ma][va1]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=250[ducked];"
                      "[va2][ducked]amix=inputs=2:duration=first:normalize=0[a]"),
                     "-map", "0:v", "-map", "[a]"])
    else:
        cmd.extend(["-filter_complex", f"[1:a]{voice_pad}[a]", "-map", "0:v", "-map", "[a]"])
    cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-t", f"{final_dur:.2f}", "-movflags", "+faststart", str(OUTPUT_PATH)])
    _run_ffmpeg(cmd)
    print(f"[VIDEO] voice={voice_dur:.1f}s final={final_dur:.1f}s → {OUTPUT_PATH}")
    return OUTPUT_PATH


# ── Upload ────────────────────────────────────────────────────────────
def _get_access_token() -> str:
    resp = requests.post(TOKEN_URL, data={
        "client_id": os.environ["YOUTUBE_CLIENT_ID"],
        "client_secret": os.environ["YOUTUBE_CLIENT_SECRET"],
        "refresh_token": os.environ["YOUTUBE_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload_video(meta: dict) -> str:
    creds = [os.getenv("YOUTUBE_CLIENT_ID"), os.getenv("YOUTUBE_CLIENT_SECRET"),
             os.getenv("YOUTUBE_REFRESH_TOKEN")]
    if not all(creds):
        print("[SKIP] Upload: missing credentials")
        return ""
    if not OUTPUT_PATH.is_file():
        print(f"[ERROR] No video at {OUTPUT_PATH}")
        return ""

    privacy = os.getenv("YOUTUBE_PRIVACY", "public")
    if privacy not in ("public", "unlisted", "private"):
        privacy = "public"

    access_token = _get_access_token()
    body = {
        "snippet": {
            "title": meta.get("title", "Reddit Stories Compilation")[:100],
            "description": meta.get("description", ""),
            "tags": meta.get("tags", _CORE_TAGS),
            "categoryId": "24",
            "defaultLanguage": "en",
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False, "embeddable": True},
    }

    video_data = OUTPUT_PATH.read_bytes()
    init_resp = requests.post(UPLOAD_URL, params={"uploadType": "resumable", "part": "snippet,status"},
                              headers={"Authorization": f"Bearer {access_token}",
                                       "Content-Type": "application/json; charset=UTF-8",
                                       "X-Upload-Content-Length": str(len(video_data)),
                                       "X-Upload-Content-Type": "video/mp4"},
                              json=body, timeout=30)
    init_resp.raise_for_status()
    upload_url = init_resp.headers["Location"]

    print(f"[UPLOAD] {len(video_data) / 1024 / 1024:.1f} MB...")
    for attempt in range(1, 4):
        try:
            resp = requests.put(upload_url, headers={"Authorization": f"Bearer {access_token}",
                                                      "Content-Type": "video/mp4",
                                                      "Content-Length": str(len(video_data))},
                                data=video_data, timeout=600)
            resp.raise_for_status()
            video_id = resp.json().get("id", "")
            print(f"[UPLOAD] https://youtube.com/watch?v={video_id}")
            try:
                from analytics import log_upload
                log_upload(video_id, meta.get("title", ""), meta.get("topic", ""), meta.get("tags", []))
            except Exception as exc:
                print(f"[WARN] Analytics: {exc}")
            return video_id
        except Exception as exc:
            print(f"[WARN] Upload attempt {attempt}: {exc}")
            if attempt < 3:
                time.sleep(attempt * 15)
    return ""


# ── Main ─────────────────────────────────────────────────────────────
def main():
    _clean_build_dir()

    print("[1/5] Generating compilation script...")
    script_data = None
    for attempt in range(3):
        script_data = generate_compilation_script()
        if script_data:
            break
        print(f"[RETRY] Attempt {attempt + 2}...")
    if not script_data:
        print("[ERROR] Failed to generate script")
        sys.exit(1)

    script_text = script_data["script"]
    meta = {
        "title": script_data.get("title", "Reddit Stories Compilation")[:100],
        "description": script_data.get("description", "") + _DESCRIPTION_FOOTER,
        "tags": list(dict.fromkeys(script_data.get("tags", []) + _CORE_TAGS))[:20],
        "topic": random.choice(COMPILATION_THEMES),
    }

    # Title dedup
    history = _load_title_history()
    title_words = set(meta["title"].lower().split())
    for old in history[-20:]:
        overlap = len(title_words & set(old.lower().split())) / max(len(title_words), 1)
        if overlap > 0.5:
            print(f"[WARN] Title too similar to recent: {old}")
            meta["title"] = f"{random.choice(['Top', 'Best', 'Wildest', 'Craziest'])} Reddit Stories #{random.randint(10,99)}"
            break
    history.append(meta["title"])
    _save_title_history(history)

    print(f"  Title: {meta['title']}")
    print(f"  Script: {len(script_text.split())} words")

    (BUILD_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[2/5] Generating voiceover...")
    audio_path, word_events = generate_tts(script_text)
    dur = _probe_duration(audio_path)
    print(f"  Duration: {dur:.1f}s ({dur/60:.1f} min)")

    print("[3/5] Downloading clips...")
    clips = download_clips(extra_queries=script_data.get("pexels_queries", []), target=35)
    if not clips:
        print("[ERROR] No clips")
        sys.exit(1)

    print("[4/5] Downloading music...")
    music = download_music()

    print("[5/5] Assembling video...")
    assemble_video(clips, audio_path, word_events, music)

    print("[UPLOAD] Uploading...")
    upload_video(meta)

    temp = BUILD_DIR / "temp"
    if temp.is_dir():
        shutil.rmtree(temp)
    print("[DONE]")


if __name__ == "__main__":
    main()
