# Deploying to Hugging Face Spaces

The app is one container: FastAPI, the static frontend, and an embedded Qdrant.
The index is built during the Docker build from `data/corpus.deploy.jsonl`, so the
running Space needs no network access to work.

---

## 1. Get the API keys

Both are optional. **The Space works with neither** — retrieval and extractive
answering are entirely local. Add them to unlock voice input and generative
answers.

### Groq — generative answers

1. Go to <https://console.groq.com> and sign in with Google or GitHub.
2. Open **API Keys** in the left sidebar → **Create API Key**.
3. Copy it immediately; the value is shown once. It starts with `gsk_`.

Groq has a genuinely free tier with no card required — that is why it is the
primary provider here. Rate limits are per-model and per-minute, which is why
`bench/run_bench.py` never calls the LLM in a tight loop.

### Sarvam — speech-to-text

1. Go to <https://dashboard.sarvam.ai> and sign up.
2. Open **API Keys** → create a key. It is a subscription key sent in the
   `api-subscription-key` header.
3. New accounts get free credits. STT is billed per hour of audio, and a demo
   uses minutes, so the free credits comfortably cover development plus judging.

> Verify current limits on both dashboards before submitting — pricing and free
> tiers change, and this file was written against what was published at build time.

### Where to put them

Locally, copy `.env.example` to `.env` and fill it in. **Never commit `.env`** —
it is gitignored. On Spaces they go in repository secrets (step 4 below), not in
the repo.

---

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Space name**: `voice-rag-msmarco-xi`
3. **License**: MIT (or your choice)
4. **SDK**: select **Docker** → **Blank**
5. **Hardware**: CPU basic (free)
6. **Visibility**: Public — judges need to open it
7. Create.

## 3. Push the code

Get a write token first: <https://huggingface.co/settings/tokens> → **New token**
→ type **Write** → copy it.

```bash
git remote add space https://huggingface.co/spaces/<your-username>/voice-rag-msmarco-xi
git push space main
```

When git asks for credentials, use your HF **username** and paste the **write
token** as the password.

If the push is rejected because the Space already has a README commit:

```bash
git pull space main --allow-unrelated-histories --no-rebase
git push space main
```

## 4. Add the keys as secrets

In the Space: **Settings** → **Variables and secrets** → **New secret**.

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your `gsk_...` key |
| `SARVAM_API_KEY` | your Sarvam subscription key |

Adding a secret restarts the Space. Secrets are injected as environment
variables, which is exactly where `harness/providers.py` and `stt/sarvam.py`
read them from — no code change needed.

## 5. Watch the build

Open the **Logs** tab. The build downloads the encoder and then builds the index;
expect several minutes. You should see the manifest printed near the end:

```
{'strategy': 'passage_native', 'passages': 18024, 'chunks': ..., 'langs': ['en', 'gu', 'hi']}
```

Then confirm it is serving:

```bash
curl https://<your-username>-voice-rag-msmarco-xi.hf.space/health
```

`indexed_points` should be non-zero and `providers` should show `configured:
true` for whichever keys you added.

---

## Things that actually break

**Microphone does nothing.** `getUserMedia` requires a secure context. Spaces
serve HTTPS so this works there and on `localhost`, but not over a bare IP.

**Build fails on permissions.** The Dockerfile creates user 1000 and runs both
the build and the runtime as that user on purpose — Spaces run the container as
uid 1000, and an index built as root leaves `.qdrant/` unwritable at startup.
Don't add a `USER root` above the build steps.

**Space sleeps.** Free Spaces pause after inactivity. Open the link a few minutes
before a demo so the first judge doesn't pay the cold start.

**`/ask-voice` returns 503.** That is the deliberate response when
`SARVAM_API_KEY` is missing. Text `/ask` still works.
