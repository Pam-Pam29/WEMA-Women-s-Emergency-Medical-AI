Superseded files from earlier development iterations (pre-final architecture). Not part of the graded system — see `src/` and `evaluation/`.

| File | Why it's here |
|---|---|
| `src/agent.py` | imports `models/risk_classifier.pkl`, which no longer exists in this repo — broken if run |
| `src/inference.py` | documents the abandoned Hugging Face Spaces + Railway deployment architecture (WEMA now runs on Fly.io) |
| `src/evaluate.py` | references `data/WEMA_Labelled_Scenarios_v1.xlsx`, which doesn't exist — superseded by `evaluation/WEMA_Testing_and_Evaluation.ipynb` |
| `Procfile` | Heroku deployment leftover |
| `railway.json` | Railway deployment leftover |
| `requirements_inference.txt` | paired with the dead `inference.py` / Hugging Face Spaces path |
| `runtime.txt` | Heroku Python-buildpack file; Fly.io ignores it (version comes from `Dockerfile`) |
| `WEMA_Data_Exploration.ipynb` | stale exploratory EDA notebook — reads an old ChromaDB snapshot with a since-superseded chunk count; no evaluation evidence, fully reproducible from the currently-committed dataset |
| `WEMA_Initial_Solution.pptx` | early solution-design deck, superseded by the current architecture and this README |
| `settings.py` | early-architecture config snapshot, never imported by anything in `src/`: names Llama 3.3 70B as the generator (it's now the evaluation judge only; Qwen3-32B is the real generator), ElevenLabs as the TTS provider (replaced by Azure Neural TTS), a `FINETUNED_MODEL_PATH` for the fine-tuning approach that was tried and abandoned in favour of RAG, and a Google Maps key / single doctor phone number (replaced by the Haversine formula over `providers.csv`) |
| `warmup.py` | pings `HF_SPACES_URL` to prevent cold starts on the abandoned Hugging Face Spaces deployment; not applicable to Fly.io, which runs with `min_machines_running = 1` and never sleeps |
