# Therapy Bot

A simple therapy bot application built with Python.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [Running the Application](#running-the-application)

## Prerequisites

- Python 3.6 or higher
- `pip` package manager

## Installation

### 1. Clone the Repository

Clone this repository to your local machine using:

```bash
git clone <repository-url>
cd <repository-directory>
```

On windows
```bash
python -m venv venv
venv\Scripts\activate
```

On macos/linux
```bash
python3 -m venv venv
source venv/bin/activate
```
Install Dependencies

```bash
pip install -r requirements.txt
```

Running the application
```bash
python therapy_bot.py
```

## Obtaining API Keys

The Therapy Bot requires API keys from **OpenAI**, **Anthropic**, and **ElevenLabs**. Follow the instructions below to obtain each key and set them up in your `.env` file.

### 1. OpenAI API Key

**Steps to Obtain:**

1. **Sign Up / Log In:**
   - If you don't have an OpenAI account, [**Sign Up Here**](https://platform.openai.com/signup).
   - If you already have an account, [**Log In Here**](https://platform.openai.com/login).

2. **Navigate to API Keys:**
   - After logging in, go to the [**API Keys**](https://platform.openai.com/account/api-keys) section in your account dashboard.

3. **Create a New API Key:**
   - Click on the **"Create new secret key"** button.
   - Copy the generated API key and store it securely. **You won't be able to view it again once you leave the page.**

### 2. Anthropic API Key

**Steps to Obtain:**

1. **Sign Up / Log In:**
   - If you don't have an Anthropic account, [**Sign Up Here**](https://www.anthropic.com/signup).
   - If you already have an account, [**Log In Here**](https://www.anthropic.com/login).

2. **Access API Section:**
   - After logging in, navigate to the **API** section from your dashboard.

3. **Generate API Key:**
   - Click on **"Generate New API Key"**.
   - Copy the generated API key and keep it secure.

> **Note:** If you encounter any issues or need access to specific API tiers, contact [Anthropic Support](https://www.anthropic.com/contact).

### 3. ElevenLabs API Key

**Steps to Obtain:**

1. **Sign Up / Log In:**
   - If you don't have an ElevenLabs account, [**Sign Up Here**](https://beta.elevenlabs.io/signup).
   - If you already have an account, [**Log In Here**](https://beta.elevenlabs.io/login).

2. **Navigate to API Keys:**
   - After logging in, go to the [**API Keys**](https://beta.elevenlabs.io/account/api-keys) section in your account settings.

3. **Create a New API Key:**
   - Click on **"Create New API Key"**.
   - Copy the generated API key and store it securely.

---

## Setting Up the `.env` File

After obtaining all the necessary API keys, you'll need to set them up in a `.env` file to configure your environment variables securely.

1. **Create a `.env` File:**

   In the root directory of your project, create a file named `.env`.

2. **Add Your API Keys:**

   Open the `.env` file in a text editor and add the following lines, replacing the placeholder text with your actual API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```