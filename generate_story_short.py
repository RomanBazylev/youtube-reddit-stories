import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import shutil

import numpy as np
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

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
MUSIC_PATH = BUILD_DIR / "music.mp3"
TITLE_HISTORY_PATH = Path("title_history.json")
USED_STORIES_PATH = Path("used_stories.json")
MAX_TITLE_HISTORY = 40  # remember last N titles to avoid repeats
MAX_USED_STORIES = 500  # remember last N Reddit post IDs to avoid repeats
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

# ── Specific diversity pools (character × setting × premise = millions of combos) ──

STORY_CHARACTERS = [
    "a retired firefighter",
    "a college freshman",
    "a single dad working two jobs",
    "a veterinarian in a small town",
    "a high school teacher",
    "a food delivery driver",
    "a 70-year-old grandmother",
    "a night shift security guard",
    "a wedding photographer",
    "a real estate agent",
    "a nurse on the night shift",
    "a first-generation college student",
    "an Uber driver",
    "a park ranger",
    "a stay-at-home mom turned entrepreneur",
    "a librarian",
    "a military veteran adjusting to civilian life",
    "a small restaurant owner",
    "a tattoo artist",
    "a foster parent",
]

STORY_SETTINGS = [
    "at a summer camp",
    "during a cross-country road trip",
    "in a hospital waiting room",
    "at a family reunion barbecue",
    "in a small-town diner",
    "at a storage unit auction",
    "during a power outage",
    "on a cruise ship",
    "at a community garage sale",
    "in a shared laundry room",
    "at a dog park",
    "during a house renovation",
    "at a 24-hour Walmart at 3 AM",
    "in a thrift store",
    "at a funeral reception",
    "during a neighborhood block party",
    "at a car dealership",
    "in an airport during a delay",
    "at a high school reunion",
    "at a pawn shop",
]

STORY_PREMISES = [
    "The narrator discovers their dog has been visiting a neighbor's house every day",
    "A package arrives addressed to someone who died 20 years ago",
    "The narrator recognizes their childhood bully working at a drive-through",
    "A security camera captures something nobody was supposed to see",
    "The narrator's identical twin impersonates them for a job interview",
    "A hidden room is found behind drywall during renovation",
    "The narrator's Uber driver turns out to be their estranged father",
    "A handwritten note is found inside a secondhand book",
    "The new neighbor's WiFi name reveals a disturbing message",
    "A DNA test kit gift reveals the family's biggest secret",
    "The narrator catches their landlord entering the apartment while they're at work",
    "A stranger at a coffee shop leaves a note saying 'I know what you did'",
    "The narrator discovers their coworker has been living in the office after hours",
    "An old voicemail is found on a phone bought at a pawn shop",
    "The narrator finds their own missing person poster from 15 years ago",
    "A storage unit left by a deceased relative contains something unexpected",
    "The narrator's Ring doorbell records the same person walking by at 3 AM every night",
    "A childhood time capsule is opened 20 years later with a shocking item inside",
    "The narrator discovers their 'online friend' of 5 years lives next door",
    "An anonymous letter arrives warning the narrator about someone they trust",
    "The narrator's kid draws a picture of 'the man in the closet'",
    "A restaurant receipt from a spouse shows a dinner for two on a 'work trip' night",
    "The narrator finds a second mailbox key on their keyring that fits a PO box they never rented",
    "A neighbor's tree falls and reveals something buried underneath",
    "The narrator gets a friend request from someone with their exact name and face",
    "A hotel room's Bible has handwritten messages from dozens of past guests telling the same warning",
    "The narrator's car dashcam records something while parked at the office",
    "A wrong-number text leads to uncovering a scam targeting the narrator's elderly parent",
    "The narrator discovers their 'new' house still has the previous owner's belongings hidden in the attic",
    "A lost wallet is returned with extra money and a note inside",
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

# ── Reddit story subreddits for infinite unique premises ───────────────
REDDIT_SUBREDDITS = [
    "tifu", "AmItheAsshole", "MaliciousCompliance", "pettyrevenge",
    "ProRevenge", "entitledparents", "IDontWorkHereLady",
    "TalesFromRetail", "TalesFromYourServer", "relationship_advice",
    "confessions", "TrueOffMyChest", "NuclearRevenge",
    "neighborsfromhell", "BestofRedditorUpdates",
]

REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def _load_used_stories() -> list:
    """Load list of previously used Reddit post IDs."""
    if USED_STORIES_PATH.is_file():
        try:
            return json.loads(USED_STORIES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_used_story(post_id: str) -> None:
    """Append Reddit post ID to used list and trim."""
    used = _load_used_stories()
    used.append(post_id)
    if len(used) > MAX_USED_STORIES:
        used = used[-MAX_USED_STORIES:]
    USED_STORIES_PATH.write_text(json.dumps(used, ensure_ascii=False), encoding="utf-8")


def fetch_reddit_premise() -> Optional[str]:
    """
    Fetch a real Reddit story from a random story subreddit.
    Returns a premise string (title + first ~300 chars of selftext),
    or None if all attempts fail.
    """
    used_ids = set(_load_used_stories())
    subreddits = list(REDDIT_SUBREDDITS)
    random.shuffle(subreddits)

    for sub in subreddits[:5]:  # try up to 5 subreddits
        for time_filter in ["week", "month", "year"]:
            url = f"https://old.reddit.com/r/{sub}/top/.json?t={time_filter}&limit=50"
            try:
                time.sleep(random.uniform(1.5, 3.5))
                resp = requests.get(
                    url,
                    headers=REDDIT_HEADERS,
                    timeout=15,
                )
                if resp.status_code == 429:
                    print(f"[REDDIT] Rate limited on r/{sub}, sleeping...")
                    time.sleep(10)
                    break
                if resp.status_code == 403:
                    print(f"[REDDIT] Blocked r/{sub}/{time_filter}, trying next...")
                    continue
                resp.raise_for_status()
                data = resp.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    p = post.get("data", {})
                    post_id = p.get("id", "")
                    selftext = (p.get("selftext") or "").strip()
                    title = (p.get("title") or "").strip()

                    # Skip: already used, too short, removed, or not a story
                    if post_id in used_ids:
                        continue
                    if len(selftext) < 200:
                        continue
                    if p.get("removed_by_category") or selftext == "[removed]" or selftext == "[deleted]":
                        continue

                    # Extract premise: title + truncated body
                    body_preview = selftext[:500].rsplit(" ", 1)[0]  # cut at word boundary
                    premise = f"{title}. {body_preview}"
                    _save_used_story(post_id)
                    print(f"[REDDIT] Got story from r/{sub}: {title[:80]}...")
                    return premise

            except Exception as exc:
                print(f"[REDDIT] Failed r/{sub}/{time_filter}: {exc}")
                continue

    print("[REDDIT] All subreddits exhausted, falling back to static premises")
    return None


@dataclass
class ScriptPart:
    text: str


@dataclass
class VideoMetadata:
    title: str
    description: str
    tags: List[str]
    topic: str = ""


@dataclass
class WordTiming:
    text: str
    offset: float
    duration: float


# ── YouTube title dedup ────────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    'shorts', 'reddit', 'the', 'a', 'my', 'i', 'and', 'to', 'for', 'of',
    'in', 'on', 'is', 'was', 'her', 'his', 'she', 'he', 'it', 'me',
})


def _title_similarity(a: str, b: str) -> float:
    """Word-overlap ratio between two titles, ignoring stop words and symbols."""
    def _norm(t):
        return set(re.sub(r'[^a-z0-9 ]', ' ', t.lower()).split()) - _STOP_WORDS
    wa, wb = _norm(a), _norm(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def get_recent_titles(limit: int = 30) -> list:
    """Fetch recent video titles from the YouTube channel via API."""
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        print("[YOUTUBE] Missing OAuth credentials, skipping title check")
        return []
    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=15,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        ch_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "contentDetails", "mine": "true"},
            headers=headers,
            timeout=15,
        )
        ch_resp.raise_for_status()
        items = ch_resp.json().get("items", [])
        if not items:
            return []
        uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        pl_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/playlistItems",
            params={
                "part": "snippet",
                "playlistId": uploads_id,
                "maxResults": str(min(limit, 50)),
            },
            headers=headers,
            timeout=15,
        )
        pl_resp.raise_for_status()
        titles = []
        for item in pl_resp.json().get("items", []):
            title = item.get("snippet", {}).get("title", "").strip()
            if title and title.lower() != "private video":
                titles.append(title)
        return titles
    except Exception as exc:
        print(f"[YOUTUBE] Could not fetch recent titles: {exc}")
        return []


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
    [
        ScriptPart("My rescue dog led me to a locked shed behind our new house."),
        ScriptPart("We adopted Max from a shelter three weeks after moving in."),
        ScriptPart("Every evening he'd scratch at the back fence and whine toward the old shed."),
        ScriptPart("I figured it was raccoons. My wife said to just ignore it."),
        ScriptPart("One Saturday I grabbed bolt cutters and opened the padlock."),
        ScriptPart("Inside were stacked boxes labeled with dates going back fifteen years."),
        ScriptPart("Every box held letters, photos, and small wrapped gifts addressed to a girl named Lily."),
        ScriptPart("I tracked down the previous owner through county records. He was in a nursing home."),
        ScriptPart("Turned out Lily was his granddaughter. Her parents cut off contact after a family fight."),
        ScriptPart("He'd been buying her birthday gifts every year, hoping she'd come back."),
        ScriptPart("I found Lily on social media. She lived two hours away and had no idea."),
        ScriptPart("She drove down that weekend. The reunion at the nursing home broke everyone in the room."),
        ScriptPart("She visits him every Sunday now. Max gets extra treats for being the one who started it all."),
        ScriptPart("Sometimes the best things you find aren't what you were looking for."),
    ],
    [
        ScriptPart("Our family vacation to Mexico turned into a survival story on night two."),
        ScriptPart("We booked an all-inclusive resort. The photos online looked like paradise."),
        ScriptPart("When we arrived, the lobby smelled like mildew and half the lights were out."),
        ScriptPart("Our room had ants in the bathroom and a balcony door that wouldn't lock."),
        ScriptPart("I complained at the front desk. The manager shrugged and said all rooms were the same."),
        ScriptPart("That night a tropical storm knocked out power to the entire resort for thirty-six hours."),
        ScriptPart("No AC, no restaurant, no working phones. Staff disappeared."),
        ScriptPart("My wife found a group of stranded tourists pooling food in the conference room."),
        ScriptPart("A retired chef from Chicago organized a meal using whatever the kitchen had left."),
        ScriptPart("Strangers became friends over canned beans and warm soda by candlelight."),
        ScriptPart("When power returned, management offered us twenty percent off our next stay."),
        ScriptPart("I posted a one-star review with photos. It went viral. The resort closed eight months later."),
        ScriptPart("The Chicago chef and I still meet up once a year. Best friendship from the worst vacation."),
    ],
    [
        ScriptPart("I went to my twenty-year high school reunion and sat next to the kid everyone bullied."),
        ScriptPart("His name was Derek. In school he wore the same three shirts and ate lunch alone."),
        ScriptPart("People called him names I won't repeat. I never joined in, but I never stopped it either."),
        ScriptPart("At the reunion he walked in wearing a tailored suit and a watch worth more than my car."),
        ScriptPart("He'd founded a cybersecurity company that was just acquired for ninety million dollars."),
        ScriptPart("The same guys who tormented him were suddenly trying to shake his hand."),
        ScriptPart("He politely declined every one. Then he sat next to me and said he remembered something."),
        ScriptPart("In tenth grade I'd left a granola bar on his desk when nobody was looking."),
        ScriptPart("I'd completely forgotten. He never did."),
        ScriptPart("He said that one small thing made him believe not everyone was cruel."),
        ScriptPart("He offered me a job that night. I started two weeks later."),
        ScriptPart("I went from a cubicle to a corner office because of a granola bar I barely remember."),
        ScriptPart("You never know which small kindness someone is holding onto for twenty years."),
    ],
    [
        ScriptPart("A hospital mixed up my blood work and told me I had six months to live."),
        ScriptPart("I was thirty-four, healthy, running half marathons every weekend."),
        ScriptPart("The doctor sat me down with a look I'll never forget and said it was stage four."),
        ScriptPart("I drove home in silence. Told my wife. She collapsed on the kitchen floor."),
        ScriptPart("I quit my job, cashed out my retirement, and took my family to every place we'd dreamed about."),
        ScriptPart("Paris. Tokyo. A cabin in the Rockies. Three months of pure presence."),
        ScriptPart("Then a nurse called. She said there'd been a mix-up with another patient's samples."),
        ScriptPart("I was completely healthy. The results belonged to someone else entirely."),
        ScriptPart("I sat on the bathroom floor and cried for an hour. Relief, anger, all of it."),
        ScriptPart("We sued. The hospital settled out of court for an amount I can't disclose."),
        ScriptPart("But here's the thing. I never went back to my old job or my old life."),
        ScriptPart("Those three months showed me what actually mattered. I started a nonprofit for patient advocacy."),
        ScriptPart("The worst phone call of my life turned into the reset I didn't know I needed."),
    ],
    # ── Stories 9-25: expanded fallback pool ──
    [
        ScriptPart("My landlord tried to evict me for having a cat. I didn't have a cat."),
        ScriptPart("The notice showed up on a Tuesday, taped to my door with a blurry photo of a tabby."),
        ScriptPart("I knocked on his door and explained I was allergic to cats and had never owned one."),
        ScriptPart("He pointed at the photo and said the camera doesn't lie."),
        ScriptPart("So I checked my Ring camera. Turns out his own cat had been sneaking into my apartment through a vent."),
        ScriptPart("I showed him the footage. His cat, Mr. Whiskers, lounging on my couch at two in the afternoon."),
        ScriptPart("He went pale. Then he tried to say it didn't matter whose cat it was."),
        ScriptPart("I called the housing authority the next morning with the footage and the eviction notice."),
        ScriptPart("Turned out he'd been pulling this scam on three other tenants to break leases early."),
        ScriptPart("He wanted to renovate and raise rents. The authorities fined him twelve thousand dollars."),
        ScriptPart("I got six months free rent as part of the settlement."),
        ScriptPart("Mr. Whiskers still visits sometimes. I bought him a little bed by the vent."),
        ScriptPart("My landlord sold the building. The new owner actually fixed the heating."),
    ],
    [
        ScriptPart("I accidentally sent a complaint about my boss to my boss."),
        ScriptPart("I was venting to my friend Lisa about how our manager Dave micromanaged every email."),
        ScriptPart("I typed out three paragraphs about his passive-aggressive meeting notes and his breath."),
        ScriptPart("Hit send. Then saw the name at the top. Dave, not Lisa. My stomach dropped."),
        ScriptPart("I tried to recall the message but Outlook said it had already been read."),
        ScriptPart("I sat at my desk for forty-five minutes waiting for the call. It never came."),
        ScriptPart("Instead, Dave walked over at lunch and said he wanted to talk privately."),
        ScriptPart("I followed him to the conference room convinced I was about to be fired."),
        ScriptPart("He closed the door, sat down, and said I was right about everything."),
        ScriptPart("He admitted he'd been stressed about his own boss micromanaging him the same way."),
        ScriptPart("He asked me to be honest about what else the team needed. I was stunned."),
        ScriptPart("That conversation changed our entire department. He loosened up, trusted us more."),
        ScriptPart("He got promoted six months later and recommended me to replace him."),
        ScriptPart("The worst email I ever sent became the best career move I ever made."),
    ],
    [
        ScriptPart("My daughter's teacher called me in for a meeting about a drawing my kid made."),
        ScriptPart("They showed me the picture. It was a person standing in fire with the caption 'Daddy's work.'"),
        ScriptPart("The principal looked at me like I was under investigation."),
        ScriptPart("I asked my daughter about it that night. She said it was exactly what she saw."),
        ScriptPart("I'm a firefighter. She drew me at work. Fighting a fire. Like I do every week."),
        ScriptPart("I brought my work ID and a photo of me in gear to the follow-up meeting."),
        ScriptPart("The principal's face turned red. The teacher was already apologizing before I sat down."),
        ScriptPart("They had almost called child services over a six-year-old's art project."),
        ScriptPart("My daughter asked me the next day if she could draw me rescuing a cat instead."),
        ScriptPart("I told her she could draw whatever she wanted. She drew me as a superhero."),
        ScriptPart("I framed that picture and put it in my locker at the station."),
        ScriptPart("The guys at work thought it was the funniest thing they'd heard all year."),
        ScriptPart("My daughter's next parent-teacher conference was a lot smoother."),
    ],
    [
        ScriptPart("I tipped a waitress a hundred dollars and got the strangest letter a week later."),
        ScriptPart("It was a Tuesday night at a diner off the highway. She looked exhausted."),
        ScriptPart("She got my order wrong twice, apologized both times, and I could see she'd been crying."),
        ScriptPart("I left a hundred on a fifteen-dollar check and wrote 'hope tomorrow is better' on the receipt."),
        ScriptPart("A week later an envelope showed up at the restaurant addressed to 'the Tuesday night man.'"),
        ScriptPart("Inside was a letter from the waitress. Her name was Maria."),
        ScriptPart("She wrote that she'd been working doubles because her car broke down and she needed it for her kid's chemo appointments."),
        ScriptPart("That tip covered the final repair payment. Her son made it to treatment on time."),
        ScriptPart("She said she almost quit that night before I walked in."),
        ScriptPart("I went back to the diner the next week. She hugged me before I could sit down."),
        ScriptPart("Her son is in remission now. I know because she sends me a Christmas card every year."),
        ScriptPart("It started with a hundred dollars and became one of the most meaningful friendships of my life."),
        ScriptPart("You never know what someone's carrying. A little kindness can change everything."),
    ],
    [
        ScriptPart("My ex showed up to my wedding. I hadn't invited her."),
        ScriptPart("She walked in during the cocktail hour wearing a white dress. My wife saw her first."),
        ScriptPart("I thought there would be a scene. Instead, my wife walked over calmly."),
        ScriptPart("She handed my ex a glass of champagne and said welcome to the party."),
        ScriptPart("My ex looked confused. She had come to make a statement. My wife took the power right out of it."),
        ScriptPart("She sat in the back during the ceremony. Nobody paid her any attention."),
        ScriptPart("At the reception my groomsman Mark asked how she got past security."),
        ScriptPart("Turns out she told them she was my cousin. They didn't check the list."),
        ScriptPart("My wife danced with me like nothing happened. She smiled the entire night."),
        ScriptPart("My ex left before the cake. She texted me the next morning and apologized."),
        ScriptPart("She said my wife's kindness made her realize she'd been holding onto anger for no reason."),
        ScriptPart("My wife never brought it up again. That's the moment I knew I had married exactly the right person."),
        ScriptPart("We've been together nine years now. She still handles chaos better than anyone I know."),
    ],
    [
        ScriptPart("I found a wallet with ten thousand dollars in it at the airport."),
        ScriptPart("It was under a seat at gate B7, no ID, just cash and a handwritten note."),
        ScriptPart("The note said: 'For Maria's surgery. Please God, let us make it in time.'"),
        ScriptPart("I turned it in to the airline desk. They said they'd try to find the owner."),
        ScriptPart("Two hours later, a man came running through the terminal in tears."),
        ScriptPart("He grabbed the wallet and just stood there shaking. He couldn't speak for a full minute."),
        ScriptPart("His daughter needed heart surgery in another city. This was every dollar they had."),
        ScriptPart("He offered me money. I told him to keep every cent."),
        ScriptPart("He asked for my address. I said just go take care of your daughter."),
        ScriptPart("Six months later I got a package in the mail from across the country."),
        ScriptPart("Inside was a drawing from a little girl named Maria. It said 'thank you for finding daddy's wallet.'"),
        ScriptPart("There was a photo of her smiling in a hospital bed, post-surgery, giving a thumbs up."),
        ScriptPart("I still have that drawing on my fridge. It reminds me that doing the right thing always matters."),
    ],
    [
        ScriptPart("My Uber driver gave me the best advice I ever received in thirty minutes."),
        ScriptPart("I had just walked out of a meeting where I got passed over for a promotion for the third time."),
        ScriptPart("I got in the car angry, probably slammed the door. He didn't say anything for five blocks."),
        ScriptPart("Then he said, you look like someone who just lost a fight they thought they'd win."),
        ScriptPart("I laughed for the first time that day. I told him everything."),
        ScriptPart("He said he used to be a VP at a tech company. Made six figures, had the corner office."),
        ScriptPart("Then he realized he was building someone else's dream while his kids grew up without him."),
        ScriptPart("He quit, started driving, and used the flexible hours to coach his son's baseball team."),
        ScriptPart("He said the promotion I wanted would just mean more hours away from the things that matter."),
        ScriptPart("He dropped me off and said one thing I'll never forget: stop climbing someone else's ladder."),
        ScriptPart("I went home, updated my resume, and applied to a smaller company closer to home."),
        ScriptPart("Got the job a month later. Less money, but I'm home for dinner every night now."),
        ScriptPart("I never got that driver's name, but I think about him every time I pick my kids up from school."),
    ],
    [
        ScriptPart("My grandma's house had a locked room nobody was allowed to enter."),
        ScriptPart("She lived alone after grandpa died. We visited every Sunday for twenty years."),
        ScriptPart("The room at the end of the hall was always locked. She said it was storage."),
        ScriptPart("When she passed at ninety-one, we found the key taped under her nightstand."),
        ScriptPart("Inside was a fully preserved art studio. Canvases everywhere, hundreds of paintings."),
        ScriptPart("Landscapes, portraits, still lifes. They were incredible. Gallery-level work."),
        ScriptPart("We had no idea she painted. Not one family member knew."),
        ScriptPart("In the back we found a letter she'd written but never sent. It was addressed to an art school."),
        ScriptPart("She had been accepted to a program in Paris in nineteen fifty-four."),
        ScriptPart("My grandfather told her it wasn't practical. She never went."),
        ScriptPart("She painted in secret for over fifty years. Every single canvas was dated."),
        ScriptPart("We donated thirty paintings to the local gallery. They sold out in a week."),
        ScriptPart("The proceeds funded an art scholarship in her name. She finally got her gallery show at ninety-one."),
        ScriptPart("Some dreams don't die. They just wait in locked rooms for someone to find them."),
    ],
    [
        ScriptPart("My coworker ate my lunch every day for two weeks. I set the perfect trap."),
        ScriptPart("I started noticing my lunch bag was lighter when I went to the fridge at noon."),
        ScriptPart("At first I thought I was imagining it. Then my entire sandwich disappeared on a Thursday."),
        ScriptPart("I asked around. Everyone shrugged. So I got creative."),
        ScriptPart("I made a beautiful-looking sandwich with Carolina Reaper hot sauce hidden under the lettuce."),
        ScriptPart("I labeled it clearly with my name and put it right in the front of the fridge."),
        ScriptPart("At twelve fifteen, I heard someone coughing violently in the break room."),
        ScriptPart("It was Greg from accounting. Red face, watery eyes, milk running down his chin."),
        ScriptPart("He looked at me and I looked at him. Nothing needed to be said."),
        ScriptPart("He never apologized, but my lunches were never touched again."),
        ScriptPart("The whole office figured out what happened within an hour."),
        ScriptPart("Someone left a bottle of Tums on Greg's desk with a sticky note that said 'bon appetit.'"),
        ScriptPart("Justice was served. Medium-rare, with a side of Carolina Reaper."),
    ],
    [
        ScriptPart("A stranger paid for my groceries and it ruined me for three days."),
        ScriptPart("I was at the self-checkout with sixty dollars worth of food and my card declined."),
        ScriptPart("I tried three times. Same result. The line behind me was growing."),
        ScriptPart("I started putting things back. The milk, the chicken, the cereal for my kids."),
        ScriptPart("A woman behind me put her hand on my arm and said don't you dare put that back."),
        ScriptPart("She swiped her card before I could argue. Sixty-two dollars and fourteen cents."),
        ScriptPart("I tried to say thank you but my voice cracked and I just stood there."),
        ScriptPart("She said she'd been exactly where I was two years ago. Someone did the same for her."),
        ScriptPart("She told me to pass it on when I could. Then she walked out."),
        ScriptPart("I sat in my car and cried for twenty minutes. Not from sadness. From being seen."),
        ScriptPart("It took me three days to stop thinking about that woman at the checkout."),
        ScriptPart("Last month I finally passed it on. A dad at Target with two kids and a maxed-out card."),
        ScriptPart("The look on his face was the same one I had. The cycle keeps going."),
    ],
    [
        ScriptPart("I discovered my neighbor had been secretly mowing my lawn for three years."),
        ScriptPart("After my wife passed, I stopped caring about the yard. About most things, honestly."),
        ScriptPart("But every Saturday morning the grass was somehow cut. I assumed the HOA did it."),
        ScriptPart("Then one day I woke up early and saw Jim from next door pushing his mower across my yard."),
        ScriptPart("He didn't know I was watching. He trimmed the edges, picked up sticks, even watered the flowers."),
        ScriptPart("I confronted him on the porch. He said he noticed I wasn't coming outside anymore."),
        ScriptPart("He said his dad went through the same thing after losing his mom. Nobody helped."),
        ScriptPart("He decided he wasn't going to let that happen to me."),
        ScriptPart("Three years. A hundred and fifty-six Saturdays. He never mentioned it once."),
        ScriptPart("I invited him in for coffee that morning. We talked for four hours."),
        ScriptPart("He told me about his dad, his divorce, his daughter in college."),
        ScriptPart("I started mowing my own lawn again after that. But every other Saturday we do it together."),
        ScriptPart("Some people don't wait to be asked. They just show up. Jim showed up for three years."),
    ],
    [
        ScriptPart("My sister's DNA test revealed our dad wasn't her biological father."),
        ScriptPart("She took it for fun. Ancestry, twenty-three and me, the whole thing."),
        ScriptPart("The results came back with zero percent match to our dad's side of the family."),
        ScriptPart("She called me panicking at eleven at night. I told her it had to be an error."),
        ScriptPart("It wasn't. I took the same test. I matched our dad. She didn't."),
        ScriptPart("We sat on this for two weeks before she decided to confront our mom."),
        ScriptPart("Mom broke down at the kitchen table. She said it happened before the marriage."),
        ScriptPart("She thought the timing worked out. She never told anyone."),
        ScriptPart("Our dad was in the next room. He walked in and said he already knew."),
        ScriptPart("He'd known since my sister was born. The blood types didn't match. He did the math."),
        ScriptPart("He said he made a choice thirty years ago to be her father anyway."),
        ScriptPart("My sister collapsed into his arms. Nobody in that kitchen had dry eyes."),
        ScriptPart("DNA doesn't make a family. Showing up every single day does. Our dad proved that."),
    ],
    [
        ScriptPart("I caught my Airbnb host watching me through a hidden camera."),
        ScriptPart("It was supposed to be a quiet weekend getaway in the mountains."),
        ScriptPart("Beautiful cabin, five stars, glowing reviews. Everything seemed perfect."),
        ScriptPart("On the second night I noticed a tiny red light blinking inside the smoke detector."),
        ScriptPart("I unscrewed it. Inside was a wireless camera pointed directly at the bed."),
        ScriptPart("I checked every room. Found two more. One in the bathroom, one in the living room."),
        ScriptPart("I didn't touch them. I took photos, packed my bags, and drove to the nearest police station."),
        ScriptPart("The detective told me they'd had complaints about that property before but never proof."),
        ScriptPart("My photos and the camera serial numbers were enough. They got a warrant within hours."),
        ScriptPart("The host had recordings of over forty guests. Stored on an external drive in his garage."),
        ScriptPart("He was arrested the next morning. It made the local news."),
        ScriptPart("Airbnb refunded me and permanently banned the listing."),
        ScriptPart("Now I check every smoke detector in every rental. Trust your instincts. That red light saved me."),
    ],
    [
        ScriptPart("My son's imaginary friend turned out to be someone real."),
        ScriptPart("He was four and kept talking about a man named George who lived in the walls."),
        ScriptPart("We thought it was adorable until he started repeating things no four-year-old would know."),
        ScriptPart("He said George told him the house used to have a blue kitchen. We had painted it white."),
        ScriptPart("Under the paint was in fact blue wallpaper. We found it during a renovation."),
        ScriptPart("I looked up the property records. The original owner's name was George Mitchell."),
        ScriptPart("He lived here from nineteen forty-two to nineteen eighty-seven. He installed the blue kitchen."),
        ScriptPart("My son described what George looked like. Short, bald, wore suspenders every day."),
        ScriptPart("I found a photo in the county archive. It matched exactly."),
        ScriptPart("I'm not a superstitious person, but I couldn't explain any of it."),
        ScriptPart("My son stopped talking about George when we finished the renovation. Just like that."),
        ScriptPart("He doesn't remember any of it now. He's twelve and thinks we're making it up."),
        ScriptPart("But I kept that county archive photo. George Mitchell is still on my fridge."),
    ],
    [
        ScriptPart("I quit my six-figure job to become a school janitor. Best decision ever."),
        ScriptPart("I was a senior analyst at a consulting firm. Seventy-hour weeks, airport lounges, hotel beds."),
        ScriptPart("My doctor said my blood pressure was borderline dangerous at thirty-nine."),
        ScriptPart("My daughter asked me one morning who I was. She was three. She didn't recognize me."),
        ScriptPart("That afternoon I typed my resignation letter. Two weeks notice, no negotiation."),
        ScriptPart("Everyone thought I lost my mind. My parents stopped talking to me for a month."),
        ScriptPart("The elementary school near our house had a janitor opening. I applied."),
        ScriptPart("The pay was a quarter of what I made. The hours were six to two. Home by lunch."),
        ScriptPart("First week, a kid spilled an entire tray of spaghetti and said sorry mister."),
        ScriptPart("I cleaned it up and he gave me a high five. That felt better than any quarterly bonus."),
        ScriptPart("I've been doing this for five years now. My blood pressure is normal. I know every kid's name."),
        ScriptPart("My daughter runs to me every morning when I drop her off. She tells her friends that's my dad."),
        ScriptPart("You can't put a price on being present. I tried. It cost me a hundred and forty thousand a year."),
        ScriptPart("Worth every penny I didn't earn."),
    ],
    [
        ScriptPart("My barber has been cutting my hair for twenty years. Last week he told me his secret."),
        ScriptPart("Every two weeks like clockwork. Same chair, same cut, same small talk."),
        ScriptPart("He never missed an appointment. Never took a vacation. I thought he just loved his work."),
        ScriptPart("Last Tuesday he sat me down and said he was retiring. Then he told me why he never stopped."),
        ScriptPart("Twenty years ago his son was diagnosed with a rare condition. Treatment wasn't covered."),
        ScriptPart("He worked six days a week, twelve hours a day, for two decades to pay it off."),
        ScriptPart("Every haircut went toward the debt. Every tip. Every holiday weekend he stayed open."),
        ScriptPart("He made the last payment three months ago. His son is healthy, married, expecting a baby."),
        ScriptPart("He showed me the zero balance statement. His hands were shaking."),
        ScriptPart("I asked why he never told anyone. He said it wasn't anyone else's burden."),
        ScriptPart("Every barber in that shop stood up and clapped when he hung up his scissors."),
        ScriptPart("I left the biggest tip of my life that day. He tried to give it back. I wouldn't let him."),
        ScriptPart("Some heroes don't wear capes. They wear aprons and hold scissors for twenty years."),
    ],
    [
        ScriptPart("My kid got suspended for punching a bully. I took him out for ice cream."),
        ScriptPart("The school called at ten AM saying my son hit another student in the cafeteria."),
        ScriptPart("I drove there expecting the worst. The principal looked at me like I'd raised a criminal."),
        ScriptPart("Then they played the security footage. A bigger kid had been shoving a girl in a wheelchair."),
        ScriptPart("He knocked her lunch tray off the table and laughed when the food hit the floor."),
        ScriptPart("My son walked over and told him to stop. The kid shoved my son. My son shoved back."),
        ScriptPart("One punch. The bully went down. Three teachers rushed over."),
        ScriptPart("They suspended my son for three days. Zero tolerance policy. The bully got the same."),
        ScriptPart("I looked at the principal and said so protecting someone in a wheelchair is the same as bullying."),
        ScriptPart("She said rules are rules. I signed the paperwork and walked out without another word."),
        ScriptPart("On the drive home my son asked if I was mad. I said we're getting ice cream."),
        ScriptPart("He lit up like it was Christmas morning. We talked about right and wrong over chocolate sundaes."),
        ScriptPart("The girl's mom called me that evening. She was in tears thanking my son."),
        ScriptPart("He's twelve and already braver than most adults I know."),
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
    VideoMetadata(
        title="My Rescue Dog Led Me to a Shed Full of Secrets 🐶 #shorts",
        description="He kept scratching at the fence every night. What was inside changed two families forever.\nSometimes the best discoveries find you.\n\n#shorts #reddit #redditstories #storytime #rescuedog #family #secrets #viral #heartwarming #reunion\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "rescue dog", "family reunion", "heartwarming", "secrets", "adopted dog", "wholesome"],
    ),
    VideoMetadata(
        title="Our Dream Vacation Turned Into a Nightmare ⛈️ #shorts",
        description="The resort looked perfect online. Then a storm knocked out everything.\nWhat happened next was something we never expected.\n\n#shorts #reddit #redditstories #storytime #vacation #travel #resort #viral #nightmare #storm\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "vacation", "travel", "resort", "storm", "nightmare", "survival"],
    ),
    VideoMetadata(
        title="The Kid Everyone Bullied Showed Up to Our Reunion a Millionaire 💼 #shorts",
        description="He wore the same three shirts in high school. Twenty years later he walked in wearing success.\nOne small act of kindness changed my life.\n\n#shorts #reddit #redditstories #storytime #reunion #bully #karma #viral #success #kindness\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "reunion", "bully", "karma", "success", "kindness", "millionaire"],
    ),
    VideoMetadata(
        title="The Hospital Said I Had 6 Months. They Were Wrong 🏥 #shorts",
        description="I quit my job, spent everything, and said goodbye. Then the phone rang.\nThe mix-up that changed my entire life.\n\n#shorts #reddit #redditstories #storytime #hospital #medical #mixup #viral #lifelesson #secondchance\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "hospital", "medical", "misdiagnosis", "second chance", "life lesson"],
    ),
    # ── Metadata for stories 9-25 ──
    VideoMetadata(
        title="My Landlord Tried to Evict Me for His Own Cat 🐱 #shorts",
        description="He taped an eviction notice on my door with a blurry photo. Turns out it was HIS cat.\nThe housing authority had a field day.\n\n#shorts #reddit #redditstories #storytime #landlord #karma #eviction #viral #drama #cat\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "landlord", "eviction", "karma", "cat", "tenant rights", "housing"],
    ),
    VideoMetadata(
        title="I Accidentally Sent My Boss a Rant About Him 💀 #shorts",
        description="Three paragraphs about his micromanagement. Sent to HIM instead of my friend.\nWhat happened next changed everything.\n\n#shorts #reddit #redditstories #storytime #boss #workplace #awkward #viral #email #office\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "boss", "workplace", "awkward moment", "email fail", "office drama", "plot twist"],
    ),
    VideoMetadata(
        title="My Kid's Drawing Almost Got Me Investigated 🔥 #shorts",
        description="The school called me in about my daughter's art. 'Daddy's work' showed a man in fire.\nI'm a firefighter. She drew me at work.\n\n#shorts #reddit #redditstories #storytime #firefighter #school #misunderstanding #viral #kids #funny\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "firefighter", "school", "misunderstanding", "kids", "funny", "parent"],
    ),
    VideoMetadata(
        title="I Tipped $100 and Got This Letter a Week Later 💌 #shorts",
        description="She got my order wrong twice and looked like she'd been crying. I left a hundred dollars.\nWhat she wrote back destroyed me.\n\n#shorts #reddit #redditstories #storytime #kindness #tip #waitress #viral #heartwarming #karma\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "kindness", "tip", "waitress", "heartwarming", "generosity", "karma"],
    ),
    VideoMetadata(
        title="My Ex Showed Up to My Wedding in White 👰 #shorts",
        description="She wasn't invited. She wore white. My wife handled it like a legend.\nThis wedding story is absolutely wild.\n\n#shorts #reddit #redditstories #storytime #wedding #ex #drama #viral #wife #class\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "wedding", "ex", "drama", "wife", "class act", "uninvited guest"],
    ),
    VideoMetadata(
        title="I Found a Wallet With $10K and a Desperate Note 💰 #shorts",
        description="Under a seat at the airport. No ID, just cash and a note about a little girl's surgery.\nWhat happened when the owner came back broke me.\n\n#shorts #reddit #redditstories #storytime #honesty #airport #viral #heartwarming #found\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "honesty", "airport", "found wallet", "heartwarming", "surgery", "good deed"],
    ),
    VideoMetadata(
        title="My Uber Driver Gave Me Life-Changing Advice 🚗 #shorts",
        description="I got passed over for promotion. He used to be a VP making six figures.\nHis advice: stop climbing someone else's ladder.\n\n#shorts #reddit #redditstories #storytime #advice #uber #career #viral #lifelesson #wisdom\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "uber driver", "advice", "career change", "life lesson", "wisdom", "inspiration"],
    ),
    VideoMetadata(
        title="Grandma's Locked Room Held a 50-Year Secret 🎨 #shorts",
        description="She said it was storage. After she passed, we found the key.\nHundreds of paintings. She'd been an artist her whole life and nobody knew.\n\n#shorts #reddit #redditstories #storytime #grandma #secret #art #viral #family #emotional\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "grandma", "secret", "art", "family", "emotional", "hidden talent", "paintings"],
    ),
    VideoMetadata(
        title="The Lunch Thief Picked the Wrong Sandwich 🌶️ #shorts",
        description="Someone ate my lunch every day for two weeks. I filled a sandwich with Carolina Reaper.\nGreg from accounting learned his lesson.\n\n#shorts #reddit #redditstories #storytime #lunchthief #revenge #office #viral #karma #spicy\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "lunch thief", "revenge", "office", "karma", "spicy", "carolina reaper", "coworker"],
    ),
    VideoMetadata(
        title="A Stranger Paid for My Groceries. I Cried in My Car 🛒 #shorts",
        description="My card declined. I started putting things back. Then a woman behind me swiped her card.\nShe'd been exactly where I was two years ago.\n\n#shorts #reddit #redditstories #storytime #kindness #groceries #payitforward #viral #emotional\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "kindness", "groceries", "pay it forward", "emotional", "stranger", "generosity"],
    ),
    VideoMetadata(
        title="My Neighbor Secretly Mowed My Lawn for 3 Years 🏡 #shorts",
        description="After my wife passed, I stopped caring about the yard. Jim from next door noticed.\n156 Saturdays. He never said a word.\n\n#shorts #reddit #redditstories #storytime #neighbor #kindness #grief #viral #wholesome #friendship\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "neighbor", "kindness", "grief", "wholesome", "friendship", "lawn", "helping"],
    ),
    VideoMetadata(
        title="A DNA Test Revealed My Dad's 30-Year Secret 🧬 #shorts",
        description="My sister's results showed zero match to our dad. We confronted our mom.\nThen dad walked in and said he already knew.\n\n#shorts #reddit #redditstories #storytime #dna #family #secret #viral #emotional #twist\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "DNA test", "family secret", "emotional", "twist", "father", "adoption"],
    ),
    VideoMetadata(
        title="I Found Hidden Cameras in My Airbnb 📷 #shorts",
        description="Beautiful cabin, five stars. Then I noticed a red light in the smoke detector.\nThree cameras. Forty victims. One arrest.\n\n#shorts #reddit #redditstories #storytime #airbnb #camera #creepy #viral #safety #crime\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "Airbnb", "hidden camera", "creepy", "safety", "crime", "travel"],
    ),
    VideoMetadata(
        title="My Son's Imaginary Friend Was a Real Person 👻 #shorts",
        description="He said George lived in the walls. He described things only the original owner would know.\nThe county archive photo matched exactly.\n\n#shorts #reddit #redditstories #storytime #paranormal #creepy #kids #viral #ghost #mystery\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "paranormal", "creepy", "imaginary friend", "ghost", "kids", "mystery", "haunted house"],
    ),
    VideoMetadata(
        title="I Quit My 6-Figure Job to Be a Janitor. Zero Regrets 🧹 #shorts",
        description="My three-year-old didn't recognize me. That afternoon I quit.\nNow I'm home by lunch every day. Worth every penny I didn't earn.\n\n#shorts #reddit #redditstories #storytime #career #quit #family #viral #lifelesson #priorities\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "career change", "quit job", "family", "janitor", "life lesson", "priorities"],
    ),
    VideoMetadata(
        title="My Barber's 20-Year Secret Made the Whole Shop Cry ✂️ #shorts",
        description="He never took a vacation in 20 years. Every haircut went to one thing.\nWhen he finally told us why, everyone stood up and clapped.\n\n#shorts #reddit #redditstories #storytime #barber #sacrifice #family #viral #emotional #hero\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "barber", "sacrifice", "family", "emotional", "hero", "dedication"],
    ),
    VideoMetadata(
        title="My Son Punched a Bully. I Took Him for Ice Cream 🍦 #shorts",
        description="He defended a girl in a wheelchair. The school suspended him.\nI signed the paperwork and drove straight to the ice cream shop.\n\n#shorts #reddit #redditstories #storytime #bully #hero #school #viral #parenting #courage\n\nFollow for a new Reddit story every day! 🔔",
        tags=["reddit", "reddit stories", "storytime", "shorts", "reddit storytime", "story time", "true story", "viral", "best reddit stories", "bully", "standing up", "school", "parenting", "courage", "hero", "wheelchair"],
    ),
]


USED_FALLBACKS_PATH = Path("used_fallbacks.json")


def _load_used_fallback_indices() -> list:
    """Load list of previously used fallback pool indices."""
    if USED_FALLBACKS_PATH.is_file():
        try:
            return json.loads(USED_FALLBACKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_used_fallback_index(idx: int) -> None:
    """Append fallback index to used list."""
    used = _load_used_fallback_indices()
    used.append(idx)
    USED_FALLBACKS_PATH.write_text(json.dumps(used, ensure_ascii=False), encoding="utf-8")


def _fallback_script(recent_titles: list = None) -> tuple:
    """Pick a fallback script, avoiding already-used stories."""
    all_titles = list(recent_titles or []) + _load_title_history()
    used_indices = set(_load_used_fallback_indices())

    # Try each story in random order, skip if index already used OR title too similar
    order = list(range(len(_FALLBACK_POOL)))
    random.shuffle(order)
    for idx in order:
        if idx in used_indices:
            continue
        meta = _FALLBACK_METADATA_POOL[idx]
        if all_titles and any(_title_similarity(meta.title, t) > 0.5 for t in all_titles):
            continue
        _save_used_fallback_index(idx)
        _save_title_to_history(meta.title)
        return list(_FALLBACK_POOL[idx]), meta

    # All fallbacks exhausted — reset tracker and pick least-recently used
    print("[WARN] All fallback stories exhausted, resetting fallback tracker")
    USED_FALLBACKS_PATH.write_text("[]", encoding="utf-8")
    idx = order[0]
    meta = _FALLBACK_METADATA_POOL[idx]
    _save_used_fallback_index(idx)
    _save_title_to_history(meta.title)
    return list(_FALLBACK_POOL[idx]), meta


def _load_title_history() -> list:
    """Load locally saved title history."""
    if TITLE_HISTORY_PATH.is_file():
        try:
            return json.loads(TITLE_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_title_to_history(title: str) -> None:
    """Append title to local history and trim to MAX_TITLE_HISTORY."""
    history = _load_title_history()
    history.append(title)
    if len(history) > MAX_TITLE_HISTORY:
        history = history[-MAX_TITLE_HISTORY:]
    TITLE_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


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


_STORY_SYSTEM_PROMPT = (
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


def _pick_story_params():
    """Pick random story parameters for LLM prompt."""
    from analytics import get_topic_weights
    weights = get_topic_weights(STORY_GENRES)
    if weights:
        genre = random.choices(STORY_GENRES, weights=weights, k=1)[0]
    else:
        genre = random.choice(STORY_GENRES)
    hook = random.choice(STORY_HOOKS)
    tone = random.choice(EMOTIONAL_TONES)
    character = random.choice(STORY_CHARACTERS)
    setting = random.choice(STORY_SETTINGS)

    reddit_premise = fetch_reddit_premise()
    if reddit_premise:
        premise = reddit_premise
        print(f"  Using Reddit premise: {premise[:100]}...")
    else:
        premise = random.choice(STORY_PREMISES)
        print(f"  Using static premise: {premise[:100]}...")

    return genre, hook, tone, character, setting, premise


def _build_user_prompt(genre, hook, tone, character, setting, premise):
    """Build the user prompt for story generation."""
    return f"""Write a complete Reddit-style story for YouTube Shorts (60–90 seconds when read aloud).

STORY PARAMETERS:
- Genre: {genre}
- Narrator: {character}
- Setting: {setting}
- Premise/Inspiration: {premise}
- Hook style: {hook}
- Emotional tone: {tone}

YOUR STORY MUST:
- Be INSPIRED by the premise above — rewrite it as a fresh, original story with different names, details, and your own creative spin
- Feature the narrator described above as the main character
- Take place in or around the setting described above
- NEVER copy text directly from the premise — adapt and transform it into something new

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


# Global for LLM-generated Pexels queries
_llm_pexels_queries: List[str] = []


def _parse_llm_response(content: str, genre: str, recent_titles: list = None):
    """Parse LLM JSON response into (parts, metadata) or None."""
    global _llm_pexels_queries
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())
    data = json.loads(content)
    parts = [ScriptPart(p["text"]) for p in data.get("parts", []) if p.get("text")]
    metadata = VideoMetadata(
        title=data.get("title", "")[:100] or "A Story You Won't Forget #shorts",
        description=data.get("description", "") or "This story has a twist you didn't see coming!\n\n#reddit #storytime #shorts",
        tags=data.get("tags", ["reddit", "storytime", "shorts"]),
        topic=genre,
    )
    metadata = _enrich_metadata(metadata)
    llm_queries = data.get("pexels_queries", [])
    if llm_queries:
        _llm_pexels_queries = [q for q in llm_queries if isinstance(q, str)][:6]

    if not _validate_script(parts):
        return None

    all_titles = list(recent_titles or []) + _load_title_history()
    if all_titles and any(_title_similarity(metadata.title, t) > 0.5 for t in all_titles):
        print(f"[WARN] Title too similar to existing: {metadata.title}")
        return None

    _save_title_to_history(metadata.title)
    return parts, metadata


def _call_gemini_for_script(recent_titles: list = None):
    """Try generating a script using Gemini 2.0 Flash as fallback LLM."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[GEMINI] GEMINI_API_KEY not set, skipping")
        return None

    genre, hook, tone, character, setting, premise = _pick_story_params()
    user_prompt = _build_user_prompt(genre, hook, tone, character, setting, premise)

    print(f"  [GEMINI] Trying Gemini 2.0 Flash...")
    print(f"  Genre: {genre}")
    print(f"  Character: {character}")
    print(f"  Setting: {setting}")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": _STORY_SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 3000},
    }

    for attempt in range(2):
        try:
            resp = requests.post(url, json=body, timeout=60)
            resp.raise_for_status()
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            result = _parse_llm_response(raw, genre, recent_titles)
            if result:
                print(f"  [GEMINI] Success on attempt {attempt + 1}")
                return result
            print(f"  [GEMINI] Attempt {attempt + 1} failed validation, retrying...")
            # Rebuild with fresh params for retry
            if attempt == 0:
                genre, hook, tone, character, setting, premise = _pick_story_params()
                user_prompt = _build_user_prompt(genre, hook, tone, character, setting, premise)
                body["contents"][0]["parts"][0]["text"] = user_prompt
                body["generationConfig"]["temperature"] = 1.0
        except Exception as exc:
            print(f"  [GEMINI] Attempt {attempt + 1} failed: {exc}")

    print("[GEMINI] All attempts failed")
    return None


def call_groq_for_script(recent_titles: list = None) -> tuple:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[WARN] GROQ_API_KEY not set — trying Gemini...")
        result = _call_gemini_for_script(recent_titles)
        if result:
            return result
        return _fallback_script(recent_titles)

    genre, hook, tone, character, setting, premise = _pick_story_params()
    user_prompt = _build_user_prompt(genre, hook, tone, character, setting, premise)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"  Genre: {genre}")
    print(f"  Character: {character}")
    print(f"  Setting: {setting}")
    print(f"  Premise: {premise}")
    print(f"  Hook: {hook}")
    print(f"  Tone: {tone}")

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": _STORY_SYSTEM_PROMPT},
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
            print(f"[WARN] Groq API attempt 2 failed: {exc2}, trying Gemini...")
            result = _call_gemini_for_script(recent_titles)
            if result:
                return result
            return _fallback_script(recent_titles)

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        result = _parse_llm_response(content, genre, recent_titles)
        if result:
            return result
        print("[WARN] Groq output failed validation, retrying with fresh prompt...")
    except Exception as exc:
        print(f"[WARN] Groq parse error, retrying: {exc}")

    # ── Retry once with a fresh random seed ──
    body["messages"][1]["content"] = user_prompt + "\n\nIMPORTANT: Use more narrative transition words like 'realized', 'discovered', 'turned out', 'finally', 'suddenly', 'that's when', 'noticed', 'decided', 'found out'. At least 30% of parts MUST contain such words."
    body["temperature"] = 1.0
    try:
        resp2 = requests.post(url, headers=headers, json=body, timeout=60)
        resp2.raise_for_status()
        content2 = resp2.json()["choices"][0]["message"]["content"]
        result2 = _parse_llm_response(content2, genre, recent_titles)
        if result2:
            return result2
        print("[WARN] Groq retry also failed validation")
    except Exception as exc:
        print(f"[WARN] Groq retry failed: {exc}")

    # ── Try Gemini before falling back to static pool ──
    print("[INFO] Groq exhausted, trying Gemini...")
    gemini_result = _call_gemini_for_script(recent_titles)
    if gemini_result:
        return gemini_result

    return _fallback_script(recent_titles)


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
        safe_msg = str(exc)
        if api_key:
            safe_msg = safe_msg.replace(api_key, "***")
        print(f"[WARN] Pixabay API error: {safe_msg}")
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


# Words that must only match as whole words (not inside "million", "bill", etc.)
_WHOLE_WORD_FIXES = {
    "SO", "MIL", "FIL", "SIL", "BIL", "OP", "GF", "BF", "DM", "PM",
    "NTA", "YTA", "ESH", "IRL",
}


# ── TTS (edge-tts, per-part) ──────────────────────────────────────────
def _fix_pronunciation(text: str) -> str:
    """Replace hard-to-pronounce abbreviations with spoken equivalents."""
    result = text
    for word, replacement in TTS_PRONUNCIATION_FIXES.items():
        if word in _WHOLE_WORD_FIXES:
            result = re.sub(r'\b' + re.escape(word) + r'\b', replacement, result, flags=re.IGNORECASE)
        else:
            result = re.sub(re.escape(word), replacement, result, flags=re.IGNORECASE)
    return result


async def _generate_part_audio(
    text: str, voice: str, rate: str, out_path: Path,
) -> List[WordTiming]:
    """Generate TTS audio and capture per-word timestamps."""
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    word_timings: List[WordTiming] = []
    audio_chunks = bytearray()

    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio_chunks.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            word_timings.append(WordTiming(
                text=chunk["text"],
                offset=chunk["offset"] / 10_000_000,
                duration=chunk["duration"] / 10_000_000,
            ))

    with out_path.open("wb") as f:
        f.write(audio_chunks)
    return word_timings


async def _generate_all_audio(
    parts: List[ScriptPart],
) -> tuple:
    """Generate per-part TTS audio with word timings. Returns (paths, timings_per_part)."""
    voice = random.choice(TTS_VOICES)
    rate = random.choice(TTS_RATE_OPTIONS)
    print(f"  TTS voice: {voice}, rate: {rate}")
    audio_paths: List[Path] = []
    all_timings: List[List[WordTiming]] = []

    for i, part in enumerate(parts):
        out = AUDIO_DIR / f"part_{i}.mp3"
        audio_paths.append(out)
        tts_text = _fix_pronunciation(part.text)
        timings = await _generate_part_audio(tts_text, voice, rate, out)
        all_timings.append(timings)

    return audio_paths, all_timings


def build_tts_per_part(parts: List[ScriptPart]) -> tuple:
    """Generate a separate mp3 for each sentence with word timings."""
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


def _make_karaoke_subtitle(
    word_timings: List[WordTiming], duration: float, is_hook: bool = False,
) -> list:
    """Karaoke-style subtitles: groups of 3 words appear in sync with speech."""
    if not word_timings:
        return []

    CHUNK_SIZE = 3
    chunks = []
    for i in range(0, len(word_timings), CHUNK_SIZE):
        chunks.append(word_timings[i:i + CHUNK_SIZE])

    layers = []
    for ci, chunk in enumerate(chunks):
        chunk_start = chunk[0].offset
        if ci + 1 < len(chunks):
            chunk_end = chunks[ci + 1][0].offset
        else:
            chunk_end = min(chunk[-1].offset + chunk[-1].duration + 0.3, duration)
        chunk_dur = chunk_end - chunk_start
        if chunk_dur <= 0:
            continue

        full_text = " ".join(w.text for w in chunk)
        active_color = "#FF4444" if is_hook else "yellow"
        active_fontsize = 88 if is_hook else 72
        fade_fontsize = 80 if is_hook else 72

        speak_end = chunk[-1].offset + chunk[-1].duration
        speak_dur = speak_end - chunk_start
        if speak_dur > 0:
            try:
                yellow_txt = (
                    TextClip(
                        full_text,
                        fontsize=active_fontsize,
                        color=active_color,
                        font="DejaVu-Sans-Bold",
                        method="caption",
                        size=(TARGET_W - 100, None),
                        stroke_color="black",
                        stroke_width=4,
                    )
                    .set_position(("center", 0.75), relative=True)
                    .set_start(chunk_start)
                    .set_duration(min(speak_dur, chunk_dur))
                )
                layers.append(yellow_txt)
            except Exception as exc:
                print(f"[WARN] Karaoke TextClip failed: {exc}")

        remaining = chunk_end - speak_end
        if remaining > 0.05:
            try:
                white_txt = (
                    TextClip(
                        full_text,
                        fontsize=fade_fontsize,
                        color="white",
                        font="DejaVu-Sans-Bold",
                        method="caption",
                        size=(TARGET_W - 100, None),
                        stroke_color="black",
                        stroke_width=3,
                    )
                    .set_position(("center", 0.75), relative=True)
                    .set_start(speak_end)
                    .set_duration(remaining)
                )
                layers.append(white_txt)
            except Exception:
                pass

    return layers


def build_video(
    parts: List[ScriptPart],
    clip_paths: List[Path],
    audio_parts: List[Path],
    music_path: Optional[Path],
    word_timings: List[List[WordTiming]],
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

        timings = word_timings[i] if i < len(word_timings) else []
        subtitle_layers = _make_karaoke_subtitle(timings, dur, is_hook=(i == 0))

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
            {"title": meta.title, "description": meta.description, "tags": meta.tags, "topic": meta.topic},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Metadata saved to {meta_path}")


def main() -> None:
    _clean_build_dir()
    ensure_dirs()
    recent_titles = get_recent_titles()
    print(f"  Found {len(recent_titles)} recent titles on channel")
    print("[1/5] Generating story script...")
    parts, metadata = call_groq_for_script(recent_titles=recent_titles)
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

    print("[3/5] Generating TTS audio (edge-tts, per-part with word timings)...")
    audio_parts, word_timings = build_tts_per_part(parts)
    for i, ap in enumerate(audio_parts):
        a = AudioFileClip(str(ap))
        wt_count = len(word_timings[i]) if i < len(word_timings) else 0
        print(f"  Part {i+1}: {a.duration:.1f}s, {wt_count} word timings")
        a.close()

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_paths, audio_parts, music_path, word_timings)
    print(f"Done! Video saved to: {output}")


if __name__ == "__main__":
    main()
