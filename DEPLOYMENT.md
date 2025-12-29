# Observa Analysis Service Deployment Guide

Python FastAPI service for ML analysis of LLM traces.

## Deployment Options

### Option 1: Railway (Recommended - Easiest)

1. **Sign up at [Railway.app](https://railway.app)**
2. **Create New Project**
3. **Deploy from GitHub:**
   - Click "New" → "GitHub Repo"
   - Select your `observa-analysis` repository
   - Railway will auto-detect Python
4. **Configure Environment Variables:**
   - Go to Settings → Variables
   - Add:
     - `PORT=8000` (Railway sets this automatically, but good to have)
     - `DATABASE_URL` (optional - if you need DB access from analysis service)
5. **Set Start Command:**
   - Settings → Deploy → Start Command:
     ```
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
6. **Deploy:**
   - Railway will automatically build and deploy
   - First deployment will take ~5-10 minutes (downloading ML models)
7. **Get Your URL:**
   - After deployment, go to Settings → Domains
   - Copy the Railway URL (e.g., `https://observa-analysis-production.up.railway.app`)

### Option 2: Render

1. **Sign up at [Render.com](https://render.com)**
2. **Create New Web Service**
3. **Connect GitHub Repository:**
   - Select `observa-analysis`
4. **Configure:**
   - **Name:** `observa-analysis`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables:**
   - Add `PORT=8000` (Render sets this automatically)
6. **Deploy:**
   - Click "Create Web Service"
   - First deployment takes ~10-15 minutes (ML models)
7. **Get Your URL:**
   - After deployment, URL will be: `https://observa-analysis.onrender.com`

### Option 3: Fly.io

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
2. **Login:**
   ```bash
   fly auth login
   ```
3. **Create App:**
   ```bash
   fly launch
   ```
4. **Configure `fly.toml`:**
   ```toml
   app = "observa-analysis"
   primary_region = "fra"

   [build]

   [http_service]
     internal_port = 8000
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
     min_machines_running = 0
     processes = ["app"]

   [[services]]
     protocol = "tcp"
     internal_port = 8000
   ```
5. **Deploy:**
   ```bash
   fly deploy
   ```

## After Deployment

1. **Test the service:**
   ```bash
   curl https://your-service-url.railway.app/health
   # Should return: {"status":"ok"}
   ```

2. **Add to observa-api:**
   - Go to Vercel Dashboard → `observa-api` project
   - Settings → Environment Variables
   - Add: `ANALYSIS_SERVICE_URL=https://your-service-url.railway.app`
   - Redeploy observa-api

3. **Verify Integration:**
   - Send a test trace to observa-api
   - Check observa-api logs for analysis service calls
   - Check analysis_results table in database

## Important Notes

- **First Deployment:** Takes 5-15 minutes (downloading ML models)
- **Model Size:** ~500MB total (DeBERTa + sentence-transformers)
- **Cold Starts:** Service may be slow on first request after idle period
- **Memory:** Needs at least 1GB RAM (Railway/Render free tier should work)
- **Cost:** Free tiers available on all platforms

## Troubleshooting

### Service Times Out
- Check logs for model download progress
- Ensure PORT environment variable is set
- Verify start command is correct

### Out of Memory
- Upgrade to higher tier (2GB+ RAM)
- Or use smaller models (modify code)

### Slow Responses
- First request after cold start is slow (loading models)
- Subsequent requests are fast
- Consider keeping service warm with health checks

## Health Check Endpoint

The service includes a health check:

```bash
GET /health
```

Returns: `{"status":"ok"}`

Use this to verify deployment and for monitoring.

