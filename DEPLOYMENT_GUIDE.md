# Stock Market Dashboard Deployment Guide

This guide will walk you through the process of deploying your Stock Market Dashboard application to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your Stock Market Dashboard code in a repository
3. A Streamlit account (free)

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "stock-market-dashboard")
4. Add a description (optional)
5. Choose public or private visibility
6. Click "Create repository"

## Step 2: Push Your Code to GitHub

From your local development environment, run the following Git commands:

```bash
# Initialize a git repository if you haven't already
git init

# Add all files to the repository
git add .

# Commit your files
git commit -m "Initial commit of Stock Market Dashboard"

# Add the remote repository
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/stock-market-dashboard.git

# Push your code to GitHub
git push -u origin main
```

Make sure your repository structure looks like this:

```
stock-market-dashboard/
├── .streamlit/
│   └── config.toml
├── app.py
├── technical_indicators.py
├── requirements.txt
└── README.md
```

## Step 3: Create a Streamlit Cloud Account

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "Sign up" or "Sign in" if you already have an account
3. Follow the prompts to create your account

## Step 4: Deploy Your App

1. From your Streamlit Cloud dashboard, click on "New app"
2. Connect your GitHub account if you haven't already
3. Select your repository, branch (usually "main"), and the main file path (app.py)
4. Add a name for your app (e.g., "Stock Market Dashboard")
5. Click "Deploy"

Streamlit Cloud will automatically detect your requirements.txt file and install the necessary dependencies.

## Step 5: Verify Your Deployment

After deployment (which may take a few minutes), your app will be available at a URL like:
```
https://yourusername-stock-market-dashboard-app-xxxx.streamlit.app
```

Visit the URL to ensure your app is working correctly.

## Troubleshooting

If you encounter any issues during deployment:

1. **App crashes during startup**: Check the logs in the Streamlit Cloud dashboard to identify the error.
2. **Missing dependencies**: Ensure all required packages are listed in your requirements.txt file.
3. **Timeout errors**: If your app takes too long to start up, you may need to optimize your code.

## Updating Your App

To update your app:

1. Make changes to your code locally
2. Commit and push to GitHub
3. Streamlit Cloud will automatically detect the changes and redeploy your app

## Advanced Configuration

For more advanced deployment options, check out the [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-cloud/get-started).

## Running the App Locally

To run the app locally, use the following command:
```
streamlit run app.py
```