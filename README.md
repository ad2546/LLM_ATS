# LLM-Powered ATS: Job Description Analysis & Resume Scoring

A Flask-based API that:
1. Accepts PDF job descriptions and uses DeepSeek (an LLM) to classify each JD into up to two real categories (e.g. “Academic Nursing”, “Nursing Research”), plus extract qualifications and requirements.  
2. Stores JDs (and their extracted fields) in a MySQL database.  
3. Iterates through a “resumes/” folder and, for each PDF resume, extracts candidate info (name, email, phone, LinkedIn, etc.) and uses DeepSeek to score the resume against the two JD categories.  
4. Persists candidates and their per-category scores in MySQL.

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation & Setup](#installation--setup)  
   - [Clone & Virtual Environment](#clone--virtual-environment)  
   - [Install Dependencies](#install-dependencies)  
   - [Environment Variables](#environment-variables)  
4. [Database Schema](#database-schema)  
   - [SQL Table Definitions](#sql-table-definitions)  
   - [Sample Data: 100 Categories](#sample-data-100-categories)  
5. [Configuration](#configuration)  
6. [Running the App](#running-the-app)  
7. [API Endpoints](#api-endpoints)  
   - [`POST /upload`](#post-upload)  
   - [`GET /recommended`](#get-recommended)  
8. [Project Structure](#project-structure)  
9. [Troubleshooting](#troubleshooting)  
10. [Contributing](#contributing)  
11. [License](#license)

---

## Features

- **PDF → Text**: uses PyMuPDF (`fitz`) to extract text from both Job Description and Resume PDFs.  
- **LLM Classification**: calls DeepSeek’s chat endpoint to classify a JD into up to two categories (pulled from your own `category` table), and to extract a one-line “qualifications” summary + one-line “requirements” summary.  
- **Resume Scoring**: for each resume PDF in `./resumes/`, extract candidate details (email, name, phone, LinkedIn) and invoke DeepSeek (once per pair of categories + JD text) to retrieve:  
  - an array of per-category `{ category, score, weight, justification }` objects,  
  - a final weighted aggregate score.  
- **MySQL Persistence**:  
  - `candidate` table stores extracted candidate info + resume path,  
  - `category_score` table stores `(candidate_id, category_name, score, weight, justification)`,  
  - `job_description` table stores JD text, detected categories (`category1_id`, `category2_id`), qualifications, requirements, …  
  - `category` table holds the master list of valid category names (e.g. “Clinical Nursing”, “Academic Nursing”, …).  
  - `logs` table captures all INFO/ERROR messages.  

---

## Prerequisites

- Python 3.9+  
- MySQL 5.7+ (or MariaDB)  
- A valid DeepSeek API key  
- Linux/macOS/Windows with Python and pip installed

---

## Installation & Setup

### Clone & Virtual Environment

```bash
# 1) Clone your GitHub repository
git clone https://github.com/ad2546/LLM_ATS.git
cd LLM_ATS

# 2) (Recommended) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # on macOS/Linux
venv\Scripts\activate.bat      # on Windows