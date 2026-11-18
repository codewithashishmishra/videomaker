# Musical Story Video App

Streamlit app that turns a prompt into a **1 second – 5 minute** cartoon video.

- **Hindi** → Indian **female** voice (`hi-IN-SwaraNeural`)
- **English** → British **female** voice (`en-GB-SoniaNeural`)
- **Content** → kids musical story **or** system-design explainer (**Mermaid.js** architecture diagrams, large designs supported — no GPT image stills / no characters)
- **Duration** → 1s … 300s (5 minutes)
- **Script** → OpenAI `gpt-4o-mini` only by default (~$0.05/run class)
- **Stills** → local `disney-pixar-cartoon` on MPS (set `USE_OPENAI_IMAGES=1` for paid OpenAI Images)
- **Static mode** → ffmpeg Ken Burns on locked stills
- **Dynamic mode** → SadTalker when a human-like face is detected; otherwise local cartoon motion (blink / tail / mouth / limbs) at 25 FPS
- **Music** → MusicGen-small (short bed + loop) with procedural fallback

## Run

```bash
cd /Users/ashishmishra/test
source venv/bin/activate
pip install -r requirements.txt
# one-time SadTalker setup (dynamic mode)
bash scripts/setup_sadtalker.sh
streamlit run app.py
```

Requires system `ffmpeg` / `ffprobe` (Homebrew: `brew install ffmpeg`).

Optional `.env`:

```
OPENAI_API_KEY=...
SADTALKER_DIR=/Users/ashishmishra/test/third_party/SadTalker
SADTALKER_PYTHON=/Users/ashishmishra/test/third_party/SadTalker/.venv/bin/python
# optional paid stills (off by default):
# USE_OPENAI_IMAGES=1
```

## Notes

- Characters are prompted with anthropomorphic Disney faces so animals/fish can still get landmarks when detection succeeds.
- Dynamic path: face gate once → SadTalker if face found; else cartoon motion (no doomed preprocess retries).
- SadTalker uses `--size 256 --batch_size 1 --still` and aborts immediately on empty face boxes.
- Legacy script: `sridha_toonworld_maker.py` (kept for reference).
