# 🧠 Guru – Data Insight Agent for Telex Platform

**Guru** is an intelligent backend agent built for the **Telex Platform (A2A ecosystem)**.  
It performs **automated data analysis, visualization, and insight generation** on structured datasets, returning meaningful summaries and artifacts that can be consumed by other agents or users within Telex.

---

## 🚀 Overview

Guru acts as the **Data Intelligence Brain** of Telex — it receives analysis requests via A2A messages, processes datasets, generates insights, and returns structured responses.

It is powered by **FastAPI**, **pandas**, **matplotlib**, and **MinIO**, with optional **spaCy** integration for natural language instruction parsing.

---

## 🧩 Key Features

- 📊 **Automated Data Insights** – correlation, regression, and statistical operations  
- 🧠 **Instruction Parsing** – understands user requests using NLP (spaCy)  
- 🧮 **Data Analysis Engine** – powered by pandas and numpy  
- 📈 **Visualizations** – bar, line, scatter, histogram charts, etc.  
- ☁️ **Artifact Storage** – chart uploads via MinIO (S3-compatible storage)  
- 🔗 **A2A Integration** – communicates seamlessly with the Telex platform  
- ⚙️ **Modular Design** – cleanly separated modules for analysis, storage, schemas, and utils  

---

## 🏗️ Architecture



User / Another Telex Agent
│
▼
Telex Platform
│
(A2A Message Exchange)
│
▼
Guru Agent (FastAPI)
│
├── NLP & Instruction Parsing (spaCy)
├── Data Analysis Engine (pandas, numpy)
├── Visualization Layer (matplotlib)
├── Storage Layer (MinIO)
└── Response Packaging (Telex-compatible schema)

Always show details

---


## 🗂️ Project Structure



data_insight_agent/
├── main.py # FastAPI entry point
├── schema.py # Pydantic models for Telex A2A message schemas
├── analysis.py # Core data analysis and visualization logic
├── utils.py # Helper utilities (e.g., regression, metadata)
├── storage/
│ └── minio_client.py # MinIO client setup and artifact upload management
└── requirements.txt # Python dependencies

Always show details

---


## ⚙️ Tech Stack

| Component | Technology | Purpose |
|------------|-------------|----------|
| **Backend Framework** | FastAPI | RESTful API and async request handling |
| **Data Processing** | pandas, numpy | Data manipulation and computation |
| **Visualization** | matplotlib | Chart and graph generation |
| **Storage** | MinIO | S3-compatible object storage for artifacts |
| **NLP Parsing** | spaCy | Instruction understanding (via `en_core_web_sm`) |
| **Packaging** | uv | Modern dependency and environment manager |

---


## 🧠 Core Workflow

1. **Receive Request**  
   Guru receives an A2A message from the Telex Platform containing:  
   - Dataset (or its URL)  
   - Analysis instruction (e.g., *“Find correlation between sales and profit”*)

2. **Parse Instruction**  
   spaCy processes the text to extract the task type and parameters.

3. **Perform Analysis**  
   The dataset is loaded into pandas, and the requested operation (e.g., correlation, regression) is executed.

4. **Visualize Results**  
   Matplotlib generates a chart or plot relevant to the analysis.

5. **Store Artifacts**  
   Charts and other outputs are uploaded to MinIO, and their URLs are recorded.

6. **Respond to Telex**  
   Guru packages results into a structured Telex-compatible response (JSON + artifacts).

---


## 🧰 Setup and Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/guru-data-insight-agent.git
cd guru-data-insight-agent

2️⃣ Install Dependencies

Using uv (recommended):

Always show details
uv sync


Or using pip:

Always show details
pip install -r requirements.txt

3️⃣ Set Up Environment Variables

Create a .env file in the project root:

Always show details
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET_NAME=guru-artifacts

4️⃣ Run the Server
Always show details
uv run uvicorn data_insight_agent.main:app --reload


Guru should now be available at:

Always show details
http://127.0.0.1:8000

🧠 Example Flow

Input (A2A message):

Always show details
{
  "instruction": "Show a regression between age and income",
  "dataset_url": "https://example.com/people.csv"
}


Output (Response to Telex):

Always show details
{
  "message": "Regression between age and income completed successfully.",
  "artifact": {
    "type": "image",
    "url": "https://minio.example.com/guru-artifacts/abc123.png"
  },
  "summary": "Income increases linearly with age up to mid-40s."
}

🧩 Example Visualization Types

bar – Category comparisons

line – Trends over time

scatter – Correlation analysis

hist – Distribution visualization

Guru uses a clean dictionary-based dispatch for visualization selection instead of repetitive conditionals.

🧱 Development Notes

Developed on Linux (WSL).

Managed via uv for environment isolation.

pipx used for global CLI tools like jupyter and uv.

All artifacts are uploaded as binary streams to MinIO.

🔮 Future Improvements

🔗 Full Telex A2A registration and handshake automation

🧠 Smarter NLP model for complex query parsing

🪄 Support for multi-dataset comparative analysis

📊 Integration with Plotly or Seaborn for richer visualizations

🧾 Insight summarization via LLM or rule-based text generation

🧑‍💻 Author

Faruq Alabi Bashir
Backend Engineer | Data Insight Developer
GitHub: @<your-username>

Email: yourname@example.com

📝 License

This project is licensed under the MIT License — you are free to use, modify, and distribute with attribution.

“Data is not just numbers — Guru helps you see the story it tells.”
