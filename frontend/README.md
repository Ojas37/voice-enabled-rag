# Voice RAG Frontend (Vercel)

This directory contains the static frontend for the **Multilingual Voice RAG Engine**, designed to be deployed to **Vercel**.

## Architecture

```
┌─────────────────────┐          ┌──────────────────────────┐
│   Vercel (Frontend) │  ──API──▶│  Render (Backend API)    │
│   frontend/         │          │  FastAPI + LanceDB + ONNX │
│   Static HTML/CSS/JS│          │  voice-rag-pipeline       │
└─────────────────────┘          └──────────────────────────┘
```

## Deploy to Vercel

### Step 1: Deploy Backend to Render first
Make sure your Render backend is live at a URL like:
`https://voice-rag-pipeline.onrender.com`

### Step 2: Deploy Frontend to Vercel

**Option A: Vercel CLI**
```bash
cd frontend
npx vercel --prod
```
When prompted, set the **Root Directory** to `frontend/`.

**Option B: Vercel Dashboard**
1. Go to [vercel.com](https://vercel.com) → New Project
2. Import `Ojas37/voice-enabled-rag` from GitHub
3. Set **Root Directory** to `frontend`
4. Add environment variable: `API_BASE_URL = https://your-render-app.onrender.com`
5. Click Deploy

### Step 3: Set Environment Variables in Vercel
In your Vercel project → Settings → Environment Variables:

| Variable | Value |
|---|---|
| `API_BASE_URL` | `https://voice-rag-pipeline.onrender.com` |

The build command (`node generate-env.js`) will inject this into `env-config.js` at deploy time.

## Local Development

To test locally, set the backend URL directly in `env-config.js`:
```js
window.__ENV__ = {
  API_BASE_URL: "http://localhost:8000"
};
```

Then open `index.html` with a local server (e.g. `npx serve .`).
