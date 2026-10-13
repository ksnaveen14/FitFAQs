# FitRAG — Deployment Guide

Three deployment targets. Pick one or deploy all three.

---

## Overview

| Platform | Cost | Sleep? | Custom Domain | Best For |
|----------|------|--------|---------------|----------|
| **Streamlit Community Cloud** | Free | Yes (after inactivity) | ✅ (via CNAME) | Portfolio, demos |
| **Railway** | Free tier / $5/mo | No | ✅ | Always-on, staging |
| **Render** | Free tier / $7/mo | Yes (free) / No (paid) | ✅ | Production-like |

> **FAISS index note:** All three platforms are stateless — the index is rebuilt from documents on each cold start. This is expected and fine for a portfolio app. For production persistence, use a hosted vector DB (Pinecone, Weaviate, Qdrant).

---

## Option 1 — Streamlit Community Cloud (Recommended for Portfolio)

**Fastest to deploy. Free. `.streamlit.app` URL looks great on a resume.**

### Steps

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "feat: FitRAG RAG app"
git remote add origin https://github.com/YOUR_USERNAME/fitrag.git
git push -u origin main
```

**2. Deploy**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click **"New app"**
- Connect your GitHub account
- Select your repo → branch: `main` → Main file: `app.py`
- Click **"Deploy"**

**3. Add your API key (critical)**
- After deploy, click **⋮ → Settings → Secrets**
- Paste exactly:
  ```toml
  OPENAI_API_KEY = "sk-your-real-key-here"
  ```
- Click **Save** — app restarts automatically

**4. Your app is live at:**
```
https://your-app-name.streamlit.app
```

### Troubleshooting
- App crashes on start → check the Streamlit Cloud logs (click "Manage app" → logs)
- `ModuleNotFoundError` → make sure `requirements.txt` is in the repo root
- API key error → verify the secret key name is exactly `OPENAI_API_KEY`

---

## Option 2 — Railway

**Always-on free tier. Docker-based. Great for staging.**

### Steps

**1. Install Railway CLI (optional but faster)**
```bash
npm install -g @railway/cli
railway login
```

**2. Deploy via CLI**
```bash
cd fitrag
railway init          # creates a new Railway project
railway up            # builds Docker image + deploys
```

**Or deploy via Dashboard (no CLI needed):**
- Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
- Select your repo — Railway auto-detects the `Dockerfile`

**3. Set environment variable**
- In Railway dashboard → your service → **Variables** tab
- Add: `OPENAI_API_KEY` = `sk-your-key-here`

**4. Generate a public URL**
- Settings → Networking → **Generate Domain**
- You get: `https://fitrag-production.up.railway.app`

**5. (Optional) Custom domain**
- Settings → Networking → Custom Domain → add your domain
- Add a CNAME record pointing to the Railway URL

### railway.toml (already in your repo)
The `railway.toml` file handles build and deploy config automatically.

### Cost
- Free tier: $5 of credit/month (enough for ~500 hours of a small container)
- Starter plan: $5/month for always-on

---

## Option 3 — Render

**Similar to Railway. Free tier sleeps after 15 min inactivity.**

### Steps

**Method A — Blueprint (one-click, recommended)**
The `render.yaml` in your repo defines everything.

- Go to [render.com](https://render.com) → New → **Blueprint**
- Connect your GitHub repo
- Render reads `render.yaml` and sets up the service automatically
- Under **Environment**, add `OPENAI_API_KEY` as a secret

**Method B — Manual**
- New → Web Service → Connect GitHub repo
- Runtime: **Docker**
- Dockerfile path: `./Dockerfile`
- Add env var: `OPENAI_API_KEY`

**Your app URL:**
```
https://fitrag.onrender.com
```

### Free vs Paid
| | Free | Starter ($7/mo) |
|--|------|-----------------|
| Sleep after inactivity | ✅ (sleeps after 15 min) | ❌ (always on) |
| Build minutes | 400/month | 400/month |
| Custom domain | ✅ | ✅ |

---

## Deploy to All Three Simultaneously

You can have the same GitHub repo deployed to all three — they're independent.

```
GitHub repo (main branch)
        │
        ├── Streamlit Cloud  →  https://fitrag.streamlit.app
        ├── Railway          →  https://fitrag.up.railway.app
        └── Render           →  https://fitrag.onrender.com
```

Each platform watches your `main` branch and auto-redeploys on `git push`.

---

## Local Docker Test (before deploying)

Test the Docker build locally before pushing to any platform:

```bash
# Build the image
docker build -t fitrag .

# Run it (replace with your actual key)
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... fitrag

# Open in browser
open http://localhost:8501
```

---

## Environment Variables Reference

| Variable | Required | Where to set |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | All platforms |

No other env vars are required. Everything else is configured via `.streamlit/config.toml`.

---

## CI/CD: Auto-deploy on Push

All three platforms auto-redeploy when you push to `main`. Your workflow:

```bash
# Make a change
git add .
git commit -m "fix: improve chunking overlap"
git push origin main
# → All connected platforms redeploy automatically
```

---

## Post-Deploy Checklist

- [ ] App loads without errors
- [ ] "Use demo documents" checkbox works (no upload needed)
- [ ] "Build Index" button runs without crashing
- [ ] A test query returns a cited answer
- [ ] RAGAS Eval tab shows metrics (takes 1-2 min)
- [ ] API key is NOT visible in logs or page source
