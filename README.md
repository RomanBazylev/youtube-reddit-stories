# Told in Secret — YouTube Shorts Auto-Generator

Automated pipeline that generates Reddit-style viral story YouTube Shorts every 4 hours using AI.

## How It Works

1. **Script Generation** — Groq LLM (llama-3.3-70b-versatile) creates a complete story with beginning, escalation, twist, and satisfying resolution
2. **Voice** — Edge TTS (en-US-GuyNeural) generates natural English narration per-sentence for perfect subtitle sync
3. **Visuals** — Atmospheric stock clips from Pexels + Pixabay with Ken Burns zoom effect
4. **Assembly** — MoviePy composites everything into a 1080×1920 vertical short (60-90 seconds)
5. **Upload** — Automatic upload to YouTube via Data API v3

## Story Quality

- **16 story genres** × **8 hook styles** × **8 emotional tones** = 1,024 unique combinations
- Quality gate ensures: minimum 12 parts, 150+ total words, narrative progression markers, satisfying endings
- Stories are told in first person with vivid details (names, times, places) for maximum engagement

## Setup

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `PEXELS_API_KEY` | Pexels API key for stock video |
| `PIXABAY_API_KEY` | Pixabay API key (optional, extra clips) |
| `YOUTUBE_CLIENT_ID` | Google OAuth2 client ID |
| `YOUTUBE_CLIENT_SECRET` | Google OAuth2 client secret |
| `YOUTUBE_REFRESH_TOKEN` | YouTube upload refresh token |

### Getting YouTube Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **YouTube Data API v3**
3. Create OAuth 2.0 credentials (Web application type)
4. Add `https://developers.google.com/oauthplayground` as authorized redirect URI
5. Use [OAuth Playground](https://developers.google.com/oauthplayground) to get refresh token with `youtube.upload` scope

## Schedule

Runs automatically every 4 hours via GitHub Actions, or trigger manually from Actions tab.
