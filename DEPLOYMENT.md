# 🚀 Deployment Guide - Render.com

This guide will help you deploy the Forest Fire Prediction project on **Render.com** for FREE.

## 📋 Prerequisites

✅ GitHub Account (already have)
✅ Render.com Account (free)
✅ Project pushed to GitHub (already done)

---

## 🎯 Step-by-Step Deployment Instructions

### Step 1: Create Render Account
1. Visit [https://render.com](https://render.com)
2. Click **"Sign up"**
3. Choose **"Sign up with GitHub"**
4. Authorize Render to access your GitHub account
5. Verify email address

---

### Step 2: Verify GitHub Files
Ensure these files exist in your repository:

✅ **Procfile** - Instructions for Render
```
web: gunicorn application:app
```

✅ **requirements.txt** - Python dependencies
```
flask==3.1.2
gunicorn==21.2.0
numpy==1.24.3
pandas==2.1.0
scikit-learn==1.8.0
matplotlib==3.8.0
seaborn==0.12.2
python-dotenv==1.0.0
```

✅ **application.py** - Flask app entry point (main app file)

✅ **templates/** - HTML templates folder
✅ **static/** - CSS, JS, images folder
✅ **models/** - Trained ML models

---

### Step 3: Update Application.py (Important!)

Make sure your `application.py` runs on the correct port for Render:

```python
import os
from flask import Flask

app = Flask(__name__)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

**Key Changes:**
- `host='0.0.0.0'` - Listen on all network interfaces
- `port` from environment variable - Render assigns the port
- `debug=False` - Production mode

---

### Step 4: Push to GitHub

Run these commands in your terminal:

```bash
# Navigate to project folder
cd "C:\myProgrammingLearning\python\projectsdatascience&mechinelearning\3-EndToEndProject"

# Stage all changes
git add .

# Commit changes
git commit -m "Add Procfile and update requirements.txt for Render deployment"

# Push to GitHub
git push origin main
```

Expected output:
```
To https://github.com/valiantProgrammer/Algerian-Forest-Fires-Prediction
   xxxx...xxxx  main -> main
```

---

### Step 5: Create Web Service on Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** (top right)
3. **Select "Web Service"**
4. **Connect Repository**:
   - Search for "Algerian-Forest-Fires-Prediction"
   - Click "Connect"
5. **Configure Web Service**:

| Setting | Value |
|---------|-------|
| **Name** | forest-fire-prediction |
| **Environment** | Python 3 |
| **Region** | Oregon (or nearest to you) |
| **Branch** | main |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn application:app` |
| **Plan** | Free |

6. **Click "Create Web Service"**

---

### Step 6: Wait for Deployment

🔄 Render will:
1. Clone your GitHub repository
2. Install dependencies from requirements.txt
3. Start your Flask app with gunicorn
4. Assign a public URL (like `https://forest-fire-prediction-xxxxx.onrender.com`)

⏱️ **Time**: Usually 3-5 minutes

✅ **Deployment Complete** when you see:
```
Your service is live at https://forest-fire-prediction-xxxxx.onrender.com
```

---

### Step 7: Access Your Deployed App

1. Copy the URL from Render dashboard
2. Open in browser: `https://forest-fire-prediction-xxxxx.onrender.com`
3. You should see your Flask app! 🎉

---

## 🔧 Environment Variables (if needed)

If you need environment variables in production:

1. Go to **Web Service Settings** on Render
2. Scroll to **Environment**
3. Add variables:
```
FLASK_ENV = production
FLASK_DEBUG = False
```

4. Click **Save Changes**
5. Service will auto-restart

---

## ⚙️ Troubleshooting

### Issue: Build fails
**Solution**: Check build logs in Render dashboard
- Look for missing packages
- Update requirements.txt
- Push changes and redeploy

### Issue: App crashes on startup
**Solution**: Check runtime logs
```
RuntimeError: You must have a Flask app in FLASK_APP
```
- Ensure `application.py` has `app = Flask(__name__)`
- Ensure `if __name__ == '__main__':` block exists

### Issue: Port errors
**Solution**: Make sure app.py uses PORT env variable
```python
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

### Issue: Static files not loading
**Solution**: Make sure paths are relative
```python
# ✅ Correct
url_for('static', filename='css/style.css')

# ❌ Wrong
'/static/css/style.css'
```

### Issue: Models or data files not found
**Solution**: Ensure they're committed to GitHub
```bash
git status  # See all files
git add models/
git add static/visualizations/
git commit -m "Add model and visualization files"
git push origin main
```

---

## 📊 Render Free Tier Features

✅ **Included**:
- Free SSL/HTTPS certificate
- Auto-redeploy on GitHub push
- Auto-scaling
- 0.5 GB RAM
- Shared CPU

⏰ **Limitation**: 
- App goes to sleep after 15 minutes of inactivity
- Takes ~30 seconds to wake up on next request

---

## 🔄 Auto-Deploy from GitHub

After initial setup, every time you push to GitHub:

1. Push code: `git push origin main`
2. Render automatically detects the change
3. Rebuilds and redeploys app
4. Live within 1-2 minutes

**No manual deployment needed!** 🎉

---

## 📈 Monitoring & Logs

### View Deployment Logs:
1. Go to your Web Service on Render
2. Click **"Logs"** tab
3. See real-time logs of your app

### Common Helpful Commands:
```bash
# Test app locally before deploying
python application.py

# Check for errors
# Look at Render dashboard "Logs" section
```

---

## 🎯 Next Steps

After deployment:
1. Share your live URL: `https://forest-fire-prediction-xxxxx.onrender.com`
2. Test all features on live app
3. Monitor logs for any issues
4. Make updates and push to GitHub

---

## 💡 Tips

✅ Keep pushing to GitHub regularly
✅ Monitor logs after each deployment
✅ Test locally before pushing
✅ Use environment variables for secrets
✅ Keep requirements.txt updated

---

## 🆘 Need Help?

**Render Documentation**: https://render.com/docs
**Flask Deployment**: https://flask.palletsprojects.com/en/latest/deploying/
**Gunicorn**: https://gunicorn.org/

---

**Your Forest Fire Prediction app is now ready for the world!** 🌍🔥

