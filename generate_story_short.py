import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import shutil

import numpy as np
from PIL import Image

import edge_tts
import requests
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    TextClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
    afx,
)

# ── Constants ──────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1080, 1920
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_DIR = BUILD_DIR / "audio_parts"
MUSIC_PATH = BUILD_DIR / "music.mp3"HISTORY_PATH = BUILD_DIR / "genre_history.json"
MAX_HISTORY = 8  # remember last N genres to avoid repeats
# Voice: natural-sounding male English voices (rotated for variety)
TTS_VOICES = [
    "en-US-AndrewMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-GuyNeural",
]
# TTS rate varies slightly per video for freshness
TTS_RATE_OPTIONS = ["+0%", "+3%", "+5%", "+7%"]

# TTS pronunciation fixes
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
    "DM": "D-M",
    "PM": "P-M",
    "IRL": "in real life",
    "NTA": "not the A-hole",
    "YTA": "you're the A-hole",
    "ESH": "everyone sucks here",
    "throwaway": "throw-away",
    "subreddit": "sub-reddit",
    "r/": "the subreddit ",
}

# ── Story categories for infinite variety ──────────────────────────────

STORY_GENRES = [
    "revenge story with a satisfying payoff",
    "wholesome twist that restores faith in humanity",
    "workplace drama with an unexpected resolution",
    "family secret that changes everything",
    "neighbor conflict with a genius solution",
    "dating horror story with a plot twist",
    "entitled person gets instant karma",
    "stranger's act of kindness with lasting impact",
    "childhood mystery finally explained years later",
    "roommate nightmare with a clever escape",
    "inheritance drama with a shocking revelation",
    "wedding disaster that turned into something beautiful",
    "school bully encounter years later with ironic twist",
    "caught in a lie — the house of cards collapses",
    "malicious compliance that backfired perfectly",
    "overheard conversation that changed my life",
]

STORY_HOOKS = [
    "starts with a seemingly normal situation that escalates fast",
    "opens with the shocking ending, then rewinds to explain how we got there",
    "begins with a simple question that spirals into chaos",
    "starts calm, then one detail flips everything upside down",
    "opens with a confession the narrator has kept for years",
    "begins with 'I thought I knew my neighbor until...'",
    "starts with a bet or dare that goes horribly wrong",
    "opens with a discovery — a letter, a photo, a message",
]

EMOTIONAL_TONES = [
    "suspenseful with growing tension",
    "darkly humorous with sharp irony",
    "heartwarming with an emotional payoff",
    "creepy and unsettling",
    "bittersweet with a life lesson",
    "satisfying justice served cold",
    "shocking revelation after revelation",
    "relatable everyday situation turned absurd",
]

PEXELS_QUERIES = [
    "person thinking alone",
    "dramatic lighting face",
    "city street night",
    "dark room atmospheric",
    "person walking away",
    "mysterious door hallway",
    "rainy window mood",
    "empty room dramatic",
    "sunset silhouette person",
    "stressed person office",
    "couple arguing",
    "office meeting tense",
    "hands shaking nervous",
    "car driving night rain",
    "person reading letter shock",
    "courtroom justice",
]


# ── Genre deduplication ───────────────────────────────────────────────────

def _load_genre_history() -> list:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_genre_history(history: list) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def _pick_unique_genre() -> str:
    """Pick a genre not recently used."""
    history = _load_genre_history()
    available = [g for g in STORY_GENRES if g not in history]
    if not available:
        available = STORY_GENRES
    genre = random.choice(available)
    history.append(genre)
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    _save_genre_history(history)
    return genre


@dataclass
class ScriptPart:
    text: str


@dataclass
class VideoMetadata:
    title: str
    description: str
    tags: List[str]


def _clean_build_dir() -> None:
    """Remove previous build artifacts to save disk space."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        print("  Cleaned previous build directory")


def ensure_dirs() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# Filler phrases that make stories feel empty
_FILLER_PATTERNS = [
    "you won't believe",
    "this is crazy",
    "wait for it",
    "stay tuned",
    "like and subscribe",
    "hear me out",
    "no literally",
    "i can't even",
    "this is so",
    "it was wild",
    "long story short",
    "anyway so",
    "basically what happened was",
    "let me tell you",
]


def _validate_script(parts: List[ScriptPart]) -> bool:
    """Validate story quality. Returns True if good enough."""
    if len(parts) < 12:
        print(f"[QUALITY] Rejected: too few parts ({len(parts)}, need >=12)")
        return False

    # Average sentence length — min 12 words for proper storytelling
    avg_words = sum(len(p.text.split()) for p in parts) / len(parts)
    if avg_words < 10:
        print(f"[QUALITY] Rejected: avg words too low ({avg_words:.1f}, need >=10)")
        return False

    # Total word count — story must be substantial (60-90 seconds)
    total_words = sum(len(p.text.split()) for p in parts)
    if total_words < 150:
        print(f"[QUALITY] Rejected: total words too low ({total_words}, need >=150)")
        return False

    # Check for filler phrases
    filler_count = 0
    for part in parts:
        text_lower = part.text.lower()
        for filler in _FILLER_PATTERNS:
            if filler in text_lower:
                filler_count += 1
                print(f"[QUALITY] Filler detected: '{part.text}'")
                break
    if filler_count > 2:
        print(f"[QUALITY] Rejected: too many fillers ({filler_count})")
        return False

    # Story must have narrative markers (beginning, middle, twist, ending)
    narrative_markers = re.compile(
        r'but then|suddenly|turned out|never expected|realized|'
        r'the truth was|plot twist|that\'s when|little did|'
        r'moment|finally|discovered|revealed|confession|'
        r'secret|stunned|shocked|couldn\'t believe|'
        r'everything changed|never spoke|walked away|'
        r'to this day|lesson|karma|justice|'
        r'noticed|decided|told|found out|knew|said|'
        r'confronted|admitted|it hit me|at that moment|'
        r'asked|called|grabbed|opened|showed|'
        r'remember|started|happened|saw|heard|'
        r'looked|turned|came|went|left|took',
        re.IGNORECASE,
    )
    narrative_count = sum(1 for p in parts if narrative_markers.search(p.text))
    ratio = narrative_count / len(parts)
    if ratio < 0.15:
        print(f"[QUALITY] Rejected: not enough narrative progression ({ratio:.0%}, need >=15%)")
        return False

    # Last part should feel like an ending (reflection, lesson, resolution)
    last_text = parts[-1].text.lower()
    ending_markers = re.compile(
        r'to this day|never|since then|learned|karma|'
        r'justice|finally|still|moral|that\'s how|'
        r'and that|ever since|in the end|looking back',
        re.IGNORECASE,
    )
    if not ending_markers.search(last_text):
        print(f"[QUALITY] Warning: ending may feel incomplete, but passing anyway")

    print(f"[QUALITY] Passed: {len(parts)} parts, avg {avg_words:.1f} words, {total_words} total, {ratio:.0%} narrative")
    return True


# ── Fallback scripts (pool of stories to avoid repeats) ───────────────
_FALLBACK_POOL = [
    [
        ScriptPart("My neighbor had been stealing my packages for months, and I finally had proof."),
        ScriptPart("It started when I noticed my Amazon deliveries kept disappearing from my doorstep."),
        ScriptPart("The first time I thought it was a mistake. The second time, I got suspicious."),
        ScriptPart("By the fifth missing package, I installed a hidden camera above my front door."),
        ScriptPart("The footage showed my neighbor Karen walking over at exactly two fifteen every afternoon."),
        ScriptPart("She'd casually pick up my package, tuck it under her arm, and walk right back to her house."),
        ScriptPart("But here's where it gets interesting. Instead of confronting her, I ordered something special."),
        ScriptPart("I bought a spring-loaded glitter bomb with a built-in camera inside the box."),
        ScriptPart("The next day, I watched the live feed as Karen carried the package into her living room."),
        ScriptPart("She opened it on her white couch. Glitter absolutely everywhere. Her scream was legendary."),
        ScriptPart("She came banging on my door, covered in glitter, demanding I pay for her couch cleaning."),
        ScriptPart("I pulled up the security footage on my phone and said maybe we should call the police instead."),
        ScriptPart("Her face went completely white. She never touched my packages again."),
        ScriptPart("She moved out three months later. Karma delivered, even when my packages weren't."),
    ],
    [
        ScriptPart("I discovered my roommate had been wearing my clothes to work every single day."),
        ScriptPart("I noticed my favorite shirts smelled like cologne I didn't own."),
        ScriptPart("One morning I decided to leave early and hide in the kitchen to watch."),
        ScriptPart("At seven fifteen, Jake walked out of his room wearing my brand new jacket."),
        ScriptPart("He stood in front of the mirror, adjusted the collar, and said 'looking good' to himself."),
        ScriptPart("I confronted him right there. He turned bright red and claimed he thought it was his."),
        ScriptPart("But then I opened his closet. Empty hangers. Every single piece was mine."),
        ScriptPart("He had been doing this for four months. My entire wardrobe was in rotation."),
        ScriptPart("I told him he had two choices: replace everything or I'd tell our landlord about the lease violation."),
        ScriptPart("He showed up the next day with six shopping bags full of new clothes. For himself."),
        ScriptPart("Turned out he'd been broke and too embarrassed to admit he couldn't afford clothes."),
        ScriptPart("I felt bad, honestly. We worked out a deal where he'd do my laundry in exchange."),
        ScriptPart("He finally got a better job two months later and paid me back for everything."),
        ScriptPart("We're still roommates. He hasn't touched my closet since."),
    ],
    [
        ScriptPart("My boss fired me on a Friday. By Monday, he was begging me to come back."),
        ScriptPart("I had worked at that company for three years, building their entire inventory system from scratch."),
        ScriptPart("He called me into his office and said the company was 'going in a new direction.'"),
        ScriptPart("I asked if there was a severance package. He laughed and said 'this isn't that kind of company.'"),
        ScriptPart("I packed my desk, said goodbye to my coworkers, and walked out without a word."),
        ScriptPart("Saturday morning I got seventeen missed calls. All from the office."),
        ScriptPart("Turned out nobody else knew the admin password to the system I built."),
        ScriptPart("Their entire warehouse operation froze. Orders couldn't ship. Clients were furious."),
        ScriptPart("My boss finally called me himself, practically begging. He offered double my old salary."),
        ScriptPart("I told him I'd come back as a consultant. Two hundred dollars an hour, minimum forty hours."),
        ScriptPart("He agreed instantly. I fixed the issue in about twenty minutes."),
        ScriptPart("Then I handed him a written password recovery guide and my final invoice."),
        ScriptPart("Eight thousand dollars for one Monday morning. He never said a word."),
        ScriptPart("I started my own consulting business that week. Best firing of my life."),
    ],
    [
        ScriptPart("I caught my best friend's boyfriend on a dating app, and I had screenshots."),
        ScriptPart("Sarah and Mike had been together for two years. She thought he was the one."),
        ScriptPart("I was swiping through an app when his face popped up. Same photos, different name."),
        ScriptPart("His profile said 'single and ready to mingle.' I almost dropped my phone."),
        ScriptPart("I took screenshots of everything — his bio, his photos, even his opening messages to other girls."),
        ScriptPart("I drove to Sarah's apartment that night. She opened the door smiling. That killed me."),
        ScriptPart("I showed her the screenshots without saying a word. Her face just crumbled."),
        ScriptPart("She called Mike right there. He denied it, said someone stole his photos."),
        ScriptPart("So I showed her the messages. He'd been active that same afternoon."),
        ScriptPart("She told him to come pick up his stuff. He showed up an hour later, furious at me."),
        ScriptPart("He said I ruined his relationship. I told him he did that all by himself."),
        ScriptPart("Sarah cried for a week, but she told me finding out then saved her from something worse."),
        ScriptPart("She met someone amazing six months later. They just got engaged last month."),
        ScriptPart("Mike still messages her sometimes. She never opens them."),
    ],
]

_FALLBACK_METADATA_POOL = [
    VideoMetadata(
        title="She Found Her Husband's Secret Phone... What She Did Next 😱 #shorts",
        description="My neighbor had been stealing my packages for months. I finally set the perfect trap.\nStay until the end — the karma is REAL.\n\n#shorts #reddit #redditstories #storytime #karma #revenge #neighbor #viral #drama #packagethief\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "karma", "revenge", "neighbor", "package thief", "glitter bomb", "caught stealing", "drama", "twist ending"],
    ),
    VideoMetadata(
        title="My Roommate Wore My Clothes for 4 Months... I Set a Trap 👔 #shorts",
        description="I noticed my shirts smelled like cologne I didn't own. Then I caught him red-handed.\nThis roommate story is absolutely wild.\n\n#shorts #reddit #redditstories #storytime #roommate #caught #drama #viral #roommatenightmare\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "roommate", "caught", "drama", "roommate nightmare", "clothes", "trap", "confrontation"],
    ),
    VideoMetadata(
        title="My Boss Fired Me Friday. Monday He Begged Me Back 💰 #shorts",
        description="He laughed when I asked about severance. He wasn't laughing on Monday morning.\nBest revenge story you'll hear today.\n\n#shorts #reddit #redditstories #storytime #revenge #workplace #karma #boss #viral #quitmyjob\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "revenge", "karma", "workplace", "boss", "fired", "quit my job", "malicious compliance", "office drama"],
    ),
    VideoMetadata(
        title="I Found My Best Friend's BF on a Dating App 📱 #shorts",
        description="She thought he was the one. I had the screenshots to prove otherwise.\nSome secrets are too big to keep.\n\n#shorts #reddit #redditstories #storytime #cheating #betrayal #drama #viral #datingapp #relationship\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "cheating", "betrayal", "drama", "dating app", "relationship", "caught cheating", "best friend", "boyfriend"],
    ),
]


def _fallback_script() -> tuple:
    idx = random.randrange(len(_FALLBACK_POOL))
    return list(_FALLBACK_POOL[idx]), _FALLBACK_METADATA_POOL[idx]


# ── Core tags that must always be present ──────────────────────────────
_CORE_TAGS = [
    "shorts", "reddit", "reddit stories", "storytime", "reddit storytime",
    "story time", "true story", "viral", "best reddit stories",
]

_DESCRIPTION_FOOTER = (
    "\n\nFollow for a new Reddit story every day! 🔔"
    "\n\n#shorts #reddit #redditstories #storytime #viral #drama #truestory"
)


def _enrich_metadata(meta: VideoMetadata) -> VideoMetadata:
    """Ensure metadata has enough tags, proper title, and rich description."""
    # Title: ensure #shorts is present and there's an emoji
    title = meta.title
    if "#shorts" not in title.lower():
        title = title.rstrip() + " #shorts"
    title = title[:100]

    # Tags: merge with core tags, deduplicate, keep order
    seen = set()
    merged_tags = []
    for tag in list(meta.tags) + _CORE_TAGS:
        tag_lower = tag.lower().strip()
        if tag_lower and tag_lower not in seen:
            seen.add(tag_lower)
            merged_tags.append(tag.strip())
    # YouTube allows up to 500 chars of tags total
    tags = []
    total_len = 0
    for tag in merged_tags:
        if total_len + len(tag) + 1 > 490:
            break
        tags.append(tag)
        total_len += len(tag) + 1

    # Description: add footer if not already rich
    desc = meta.description.strip()
    if "follow" not in desc.lower() and "subscribe" not in desc.lower():
        desc = desc + _DESCRIPTION_FOOTER

    return VideoMetadata(title=title, description=desc, tags=tags)


def call_groq_for_script() -> tuple:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[WARN] GROQ_API_KEY not set — using fallback script")
        return _fallback_script()

    genre = _pick_unique_genre()
    hook = random.choice(STORY_HOOKS)
    tone = random.choice(EMOTIONAL_TONES)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a master storyteller who writes viral Reddit-style stories for YouTube Shorts. "
        "Your stories are COMPLETE narratives with a clear beginning, rising tension, a twist or climax, and a SATISFYING ending. "
        "Every story must feel FINISHED — the listener should feel closure, not like it was cut short. "
        "Write in first person as if telling a true personal experience. "
        "Use vivid details, specific names, times, places to make the story feel real. "
        "Build suspense naturally — each sentence should make the listener NEED to hear the next one. "
        "The ending must deliver: either justice, karma, a twist reveal, an emotional payoff, or a powerful life lesson. "
        "NEVER use filler phrases like 'you won't believe' or 'wait for it'. SHOW, don't tell. "
        "Use NARRATIVE TRANSITIONS to move the story forward — words and phrases like: "
        "'but then', 'suddenly', 'turned out', 'realized', 'discovered', 'finally', "
        "'that's when', 'never expected', 'the truth was', 'everything changed', "
        "'couldn't believe', 'noticed', 'decided', 'told', 'found out', 'knew', "
        "'walked away', 'confronted', 'admitted', 'it hit me', 'at that moment'. "
        "At least a third of your sentences should contain such narrative progression markers. "
        "Respond ONLY with valid JSON, no markdown wrappers or explanations."
    )

    user_prompt = f"""Write a complete Reddit-style story for YouTube Shorts (60–90 seconds when read aloud).

STORY PARAMETERS:
- Genre: {genre}
- Hook style: {hook}
- Emotional tone: {tone}

CRITICAL STORY REQUIREMENTS:
1. OPENING (parts 1-2): A gripping hook that immediately creates curiosity or tension. Drop the listener right into the situation.
2. SETUP (parts 3-5): Establish the specific situation with vivid details — names, places, times. Make it feel REAL.
3. ESCALATION (parts 6-9): Build tension. Each sentence raises the stakes. Add complications, discoveries, or confrontations.
4. CLIMAX/TWIST (parts 10-12): The payoff moment. A revelation, a confrontation, karma, or an unexpected turn.
5. RESOLUTION (parts 13-15): A SATISFYING conclusion. The listener must feel the story is COMPLETE. End with consequence, reflection, or lasting impact.

STYLE RULES:
- First person narration, conversational tone, as if telling a friend
- Each part = 1-2 sentences, 12-25 words. Long enough for substance, short enough for pacing.
- Use specific details: "My neighbor Karen", "every Tuesday at 3 PM", "a 1997 Honda Civic"
- Natural dialogue snippets make stories feel alive: He said, "You're fired." I said, "Actually, check your email."
- NO filler: never say "you won't believe", "wait for it", "this is crazy", "hear me out"
- Use NARRATIVE TRANSITIONS in at least 30% of lines — words like: "but then", "suddenly", "turned out", "realized", "discovered", "finally", "that's when", "never expected", "couldn't believe", "noticed", "decided", "found out", "knew", "confronted", "admitted". These transitions move the story forward and keep viewers hooked.
- 14-18 parts total for a 60-90 second story
- The LAST part must feel like a definitive ending — not a cliffhanger

EXAMPLE OF A GOOD ENDING: "She moved out three months later. To this day, she crosses the street when she sees me."
EXAMPLE OF A BAD ENDING: "And that's my story. Like and subscribe for more!"

Format — strictly JSON:
{{
  "title": "Catchy clickbait YouTube title, max 80 chars. MUST include one emoji and end with #shorts. Use curiosity gap: 'She Found His Secret Phone...' or 'My Boss Fired Me. He Regretted It Monday.' Make viewers NEED to click.",
  "description": "4-6 line YouTube description. Line 1: a hook that creates curiosity (this shows in search results). Line 2: one-sentence story teaser. Line 3: empty line. Line 4-5: relevant hashtags (start with #shorts #reddit #storytime then add 5-8 story-specific hashtags like #revenge #karma #cheating #workplace #drama #betrayal #twist #confession). Line 6: call to action like 'Follow for daily Reddit stories!'",
  "tags": ["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", ...8-12 MORE story-specific tags like: "revenge", "karma", "cheating story", "workplace drama", "plot twist", "relationship", "caught cheating", "entitled people", "AITA", "best reddit stories", "reddit readings"],
  "pexels_queries": ["4-6 short English queries for atmospheric/moody stock video clips matching the story mood"],
  "parts": [
    {{ "text": "Story sentence, 12-25 words, vivid and specific" }}
  ]
}}"""

    print(f"  Genre: {genre}")
    print(f"  Hook: {hook}")
    print(f"  Tone: {tone}")

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 3000,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Groq API attempt 1 failed: {exc}, retrying...")
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
        except Exception as exc2:
            print(f"[WARN] Groq API attempt 2 failed: {exc2}, using fallback")
            return _fallback_script()

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())
        data = json.loads(content)
        parts = [ScriptPart(p["text"]) for p in data.get("parts", []) if p.get("text")]
        metadata = VideoMetadata(
            title=data.get("title", "")[:100] or "A Story You Won't Forget #shorts",
            description=data.get("description", "") or "This story has a twist you didn't see coming!\n\n#reddit #storytime #shorts",
            tags=data.get("tags", ["reddit", "storytime", "shorts"]),
        )
        metadata = _enrich_metadata(metadata)
        # Save LLM-generated Pexels queries
        llm_queries = data.get("pexels_queries", [])
        if llm_queries:
            global _llm_pexels_queries
            _llm_pexels_queries = [q for q in llm_queries if isinstance(q, str)][:6]

        if _validate_script(parts):
            return parts, metadata
        print("[WARN] LLM output failed quality check, retrying with fresh prompt...")
    except Exception as exc:
        print(f"[WARN] Groq parse error, retrying: {exc}")

    # ── Retry once with a fresh random seed ──
    body["messages"][1]["content"] = body["messages"][1]["content"] + "\n\nIMPORTANT: Use more narrative transition words like 'realized', 'discovered', 'turned out', 'finally', 'suddenly', 'that's when', 'noticed', 'decided', 'found out'. At least 30% of parts MUST contain such words."
    body["temperature"] = 1.0
    try:
        resp2 = requests.post(url, headers=headers, json=body, timeout=60)
        resp2.raise_for_status()
        content2 = resp2.json()["choices"][0]["message"]["content"]
        content2 = re.sub(r"^```(?:json)?\s*", "", content2.strip())
        content2 = re.sub(r"\s*```$", "", content2.strip())
        data2 = json.loads(content2)
        parts2 = [ScriptPart(p["text"]) for p in data2.get("parts", []) if p.get("text")]
        metadata2 = VideoMetadata(
            title=data2.get("title", "")[:100] or "A Story You Won't Forget #shorts",
            description=data2.get("description", "") or "This story has a twist you didn't see coming!\n\n#reddit #storytime #shorts",
            tags=data2.get("tags", ["reddit", "storytime", "shorts"]),
        )
        metadata2 = _enrich_metadata(metadata2)
        llm_queries2 = data2.get("pexels_queries", [])
        if llm_queries2:
            _llm_pexels_queries = [q for q in llm_queries2 if isinstance(q, str)][:6]
        if _validate_script(parts2):
            return parts2, metadata2
        print("[WARN] Retry also failed quality check, using fallback")
    except Exception as exc:
        print(f"[WARN] Retry failed: {exc}, using fallback")

    return _fallback_script()


# Global for LLM-generated Pexels queries
_llm_pexels_queries: List[str] = []


# ── Download clips ─────────────────────────────────────────────────────
def _download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _pexels_best_file(video_files: list) -> Optional[dict]:
    """Pick the best HD file from Pexels video_files list."""
    hd = [f for f in video_files if (f.get("height") or 0) >= 720]
    if hd:
        return min(hd, key=lambda f: abs((f.get("height") or 0) - 1920))
    if video_files:
        return max(video_files, key=lambda f: f.get("height") or 0)
    return None


def download_pexels_clips(target_count: int = 18) -> List[Path]:
    """Download clips using LLM-generated + fallback queries for visual diversity."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    headers = {"Authorization": api_key}
    all_queries = list(_llm_pexels_queries)
    extra = [q for q in PEXELS_QUERIES if q not in all_queries]
    random.shuffle(extra)
    all_queries.extend(extra)
    queries = all_queries[:target_count]
    result_paths: List[Path] = []
    seen_ids: set = set()
    clip_idx = 0

    for query in queries:
        if len(result_paths) >= target_count:
            break
        params = {
            "query": query,
            "per_page": 3,
            "orientation": "portrait",
        }
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers, params=params, timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Pexels search '{query}' failed: {exc}")
            continue

        for video in resp.json().get("videos", []):
            vid_id = video.get("id")
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)
            best = _pexels_best_file(video.get("video_files", []))
            if not best:
                continue
            clip_idx += 1
            clip_path = CLIPS_DIR / f"pexels_{clip_idx}.mp4"
            try:
                _download_file(best["link"], clip_path)
                result_paths.append(clip_path)
                print(f"    Pexels [{query}] -> clip {clip_idx}")
            except Exception as exc:
                print(f"[WARN] Pexels clip {clip_idx} download failed: {exc}")
            if len(result_paths) >= target_count:
                break

    return result_paths


def download_pixabay_clips(max_clips: int = 4) -> List[Path]:
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return []

    params = {
        "key": api_key,
        "q": random.choice(_llm_pexels_queries or ["dark mood", "dramatic person", "night city"]),
        "per_page": max_clips,
        "safesearch": "true",
        "order": "popular",
    }

    try:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Pixabay API error: {exc}")
        return []

    data = resp.json()
    result_paths: List[Path] = []

    for idx, hit in enumerate(data.get("hits", [])[:max_clips], start=1):
        videos = hit.get("videos") or {}
        cand = videos.get("large") or videos.get("medium") or videos.get("small")
        if not cand or "url" not in cand:
            continue
        url = cand["url"]
        clip_path = CLIPS_DIR / f"pixabay_{idx}.mp4"
        try:
            _download_file(url, clip_path)
            result_paths.append(clip_path)
        except Exception as exc:
            print(f"[WARN] Failed to download Pixabay clip {idx}: {exc}")

    return result_paths


def download_background_music() -> Optional[Path]:
    """Download atmospheric/dark background music for storytelling."""
    if os.getenv("DISABLE_BG_MUSIC") == "1":
        return None

    candidate_urls = [
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Its_time_for_adventure/Komiku_-_05_-_Friends.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Daydream/Podington_Bear_-_Daydream.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Lobo_Loco/Folkish_things/Lobo_Loco_-_01_-_Acoustic_Dreams_ID_1199.mp3",
    ]

    # Pick a random track each time for variety
    for url in random.sample(candidate_urls, len(candidate_urls)):
        try:
            _download_file(url, MUSIC_PATH)
            return MUSIC_PATH
        except Exception:
            continue
    return None


# ── TTS (edge-tts, per-part) ──────────────────────────────────────────
def _fix_pronunciation(text: str) -> str:
    """Replace hard-to-pronounce abbreviations with spoken equivalents."""
    result = text
    for word, replacement in TTS_PRONUNCIATION_FIXES.items():
        result = re.sub(re.escape(word), replacement, result, flags=re.IGNORECASE)
    return result


async def _generate_all_audio(parts: List[ScriptPart]) -> List[Path]:
    """Generate all audio parts in parallel."""
    voice = random.choice(TTS_VOICES)
    rate = random.choice(TTS_RATE_OPTIONS)
    print(f"  TTS voice: {voice}, rate: {rate}")
    audio_paths: List[Path] = []
    tasks = []
    for i, part in enumerate(parts):
        out = AUDIO_DIR / f"part_{i}.mp3"
        audio_paths.append(out)
        tts_text = _fix_pronunciation(part.text)
        comm = edge_tts.Communicate(tts_text, voice, rate=rate)
        tasks.append(comm.save(str(out)))
    await asyncio.gather(*tasks)
    return audio_paths


def build_tts_per_part(parts: List[ScriptPart]) -> List[Path]:
    """Generate a separate mp3 for each sentence — perfect sync."""
    return asyncio.run(_generate_all_audio(parts))


# ── Video assembly ────────────────────────────────────────────────────
def _fit_clip_to_frame(clip: VideoFileClip, duration: float) -> VideoFileClip:
    """Trim/loop clip to target duration and crop to 9:16."""
    if clip.duration > duration + 0.5:
        max_start = clip.duration - duration
        start = random.uniform(0, max_start)
        segment = clip.subclip(start, start + duration)
    else:
        segment = clip.fx(vfx.loop, duration=duration)

    margin = 1.10
    src_ratio = segment.w / segment.h
    target_ratio = TARGET_W / TARGET_H
    if src_ratio > target_ratio:
        segment = segment.resize(height=int(TARGET_H * margin))
    else:
        segment = segment.resize(width=int(TARGET_W * margin))

    segment = segment.crop(
        x_center=segment.w / 2, y_center=segment.h / 2,
        width=TARGET_W, height=TARGET_H,
    )
    return segment


def _apply_ken_burns(clip, duration: float):
    """Slow zoom for cinematic feel."""
    direction = random.choice(["in", "out"])
    start_scale = 1.0
    end_scale = random.uniform(1.04, 1.09)  # Subtler zoom for stories
    if direction == "out":
        start_scale, end_scale = end_scale, start_scale

    def make_frame(get_frame, t):
        progress = t / max(duration, 0.01)
        scale = start_scale + (end_scale - start_scale) * progress
        frame = get_frame(t)
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        return arr[y_off:y_off + h, x_off:x_off + w]

    return clip.fl(make_frame)


def _make_subtitle(text: str, duration: float) -> list:
    """Subtitle with outline — readable on any background."""
    shadow = (
        TextClip(
            text,
            fontsize=72,
            color="black",
            font="DejaVu-Sans-Bold",
            method="caption",
            size=(TARGET_W - 90, None),
            stroke_color="black",
            stroke_width=6,
        )
        .set_position(("center", 0.70), relative=True)
        .set_duration(duration)
    )
    main_txt = (
        TextClip(
            text,
            fontsize=72,
            color="white",
            font="DejaVu-Sans-Bold",
            method="caption",
            size=(TARGET_W - 90, None),
            stroke_color="black",
            stroke_width=3,
        )
        .set_position(("center", 0.70), relative=True)
        .set_duration(duration)
    )
    return [shadow, main_txt]


def build_video(
    parts: List[ScriptPart],
    clip_paths: List[Path],
    audio_parts: List[Path],
    music_path: Optional[Path],
) -> Path:
    if not clip_paths:
        raise RuntimeError("No video clips downloaded. Provide PEXELS_API_KEY or PIXABAY_API_KEY.")

    part_audios = [AudioFileClip(str(p)) for p in audio_parts]
    durations = [a.duration for a in part_audios]
    total_duration = sum(durations)

    voice = concatenate_audioclips(part_audios)

    if len(clip_paths) >= len(parts):
        chosen_clips = random.sample(clip_paths, len(parts))
    else:
        chosen_clips = clip_paths[:]
        random.shuffle(chosen_clips)
        while len(chosen_clips) < len(parts):
            chosen_clips.append(random.choice(clip_paths))

    source_clips = []
    video_clips = []
    for i, part in enumerate(parts):
        src_path = chosen_clips[i]
        clip = VideoFileClip(str(src_path))
        source_clips.append(clip)
        dur = durations[i]

        fitted = _fit_clip_to_frame(clip, dur)
        fitted = _apply_ken_burns(fitted, dur)

        subtitle_layers = _make_subtitle(part.text, dur)

        composed = CompositeVideoClip(
            [fitted] + subtitle_layers,
            size=(TARGET_W, TARGET_H),
        ).set_duration(dur)
        video_clips.append(composed)

    # Smooth fade-in for each clip except the first
    FADE_DUR = 0.3
    for idx in range(1, len(video_clips)):
        video_clips[idx] = video_clips[idx].crossfadein(FADE_DUR)

    video = concatenate_videoclips(video_clips, method="compose").set_duration(total_duration)

    # Audio: voice + quiet atmospheric music
    audio_tracks = [voice]
    bg = None
    if music_path and music_path.is_file():
        bg = AudioFileClip(str(music_path)).volumex(0.13)
        bg = bg.set_duration(total_duration)
        bg = bg.fx(afx.audio_fadeout, min(2.0, total_duration * 0.1))
        audio_tracks.append(bg)

    final_audio = CompositeAudioClip(audio_tracks)
    video = video.set_audio(final_audio).set_duration(total_duration)

    output_path = BUILD_DIR / "output_story_short.mp4"
    video.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="8000k",
        threads=4,
    )

    # Properly close all resources
    voice.close()
    if bg is not None:
        bg.close()
    for a in part_audios:
        a.close()
    for vc in video_clips:
        vc.close()
    for sc in source_clips:
        sc.close()
    video.close()

    return output_path


def _save_metadata(meta: VideoMetadata) -> None:
    """Save video metadata to JSON for upload step."""
    meta_path = BUILD_DIR / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {"title": meta.title, "description": meta.description, "tags": meta.tags},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Metadata saved to {meta_path}")


def main() -> None:
    _clean_build_dir()
    ensure_dirs()
    print("[1/5] Generating story script...")
    parts, metadata = call_groq_for_script()
    print(f"  Script: {len(parts)} parts")
    print(f"  Title: {metadata.title}")
    total_words = 0
    for i, p in enumerate(parts, 1):
        wc = len(p.text.split())
        total_words += wc
        print(f"  [{i}] ({wc}w) {p.text}")
    est_duration = total_words / 2.5  # ~2.5 words/sec for English TTS
    print(f"  Estimated duration: ~{est_duration:.0f}s ({total_words} words)")
    _save_metadata(metadata)

    print("[2/5] Downloading video clips...")
    clip_paths = download_pexels_clips()
    clip_paths += download_pixabay_clips()
    print(f"  Downloaded {len(clip_paths)} clips")

    print("[3/5] Generating TTS audio (edge-tts, per-part)...")
    audio_parts = build_tts_per_part(parts)
    for i, ap in enumerate(audio_parts):
        a = AudioFileClip(str(ap))
        print(f"  Part {i+1}: {a.duration:.1f}s")
        a.close()

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_paths, audio_parts, music_path)
    print(f"Done! Video saved to: {output}")


if __name__ == "__main__":
    main()
