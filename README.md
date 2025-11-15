# 🧠 **Guru – Data Insight Agent for the Telex Platform**

**Guru** is an intelligent backend agent built for the **Telex Platform (A2A ecosystem)**.
It performs **automated data analysis, visualization, instruction interpretation, and insight generation** on structured datasets, returning meaningful summaries and artifacts that can be consumed by other agents or users within Telex.

---

## 🚀 **Overview**

Guru acts as the **Data Intelligence Brain** of Telex — receiving analysis requests via A2A messages, interpreting user instructions with both NLP and LLMs, processing datasets, generating insights, and returning rich, structured responses.

It is powered by **FastAPI**, **pandas**, **matplotlib**, **MinIO**, **spaCy**, and an **LLM model: Qwen2.5-7B** for high-level instruction understanding.

---

## 🧩 **Key Features**

* 📊 **Automated Data Insights** — correlation, regression, quantiles, summary statistics, and more
* 🧠 **Instruction Parsing (NLP)** — spaCy + custom logic to extract intent and parameters
* 🤖 **Instruction Interpretation (LLM)** — Qwen2.5-7B with strict schema-constrained prompting
* 🧮 **Data Analysis Engine** — pandas + numpy
* 📈 **Visualizations** — bar, line, scatter, histogram, and more using matplotlib
* ☁️ **Artifact Storage** — MinIO (S3-compatible), with automatic upload and URL generation
* 🔗 **A2A Integration** — seamless communication within the Telex platform
* 🧱 **Modular Design** — clean architecture with separation between analysis, schema, storage, and LLM logic

---

## 🏗️ **Architecture**

```
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
           ├── Instruction Interpretation (Qwen2.5-7B)
           ├── Data Analysis Engine (pandas, numpy)
           ├── Visualization Layer (matplotlib)
           ├── Storage Layer (MinIO)
           └── Response Packaging (Telex-compatible schemas)
```

---

## 🗂️ **Project Structure**

```
data_insight_agent/
├── main.py              # FastAPI entry point
├── schema.py            # Pydantic models for Telex A2A message formats
├── analysis.py          # Core analysis and visualization logic
├── utils.py             # Helper utilities (regression, metadata extraction)
├── minio_client.py      # MinIO client + artifact upload
├── prompt.py            # LLM prompt templates and schema definitions
├── ollama_client.py     # Ollama client + model interactions
└── requirements.txt     # Project dependencies
```

---

## ⚙️ **Tech Stack**

| Component               | Technology                           | Purpose                                          |
| ----------------------- | ------------------------------------ | ------------------------------------------------ |
| **Backend Framework**   | FastAPI, Pydantic, Pydantic-Settings | API + validation + configuration                 |
| **Data Processing**     | pandas, numpy                        | Data manipulation and computation                |
| **Visualization**       | matplotlib                           | Plot and chart generation                        |
| **LLM / AI Layer**      | Qwen2.5-7B (via Ollama)              | Instruction interpretation, structured reasoning |
| **NLP Parsing**         | spaCy (`en_core_web_sm`)             | Intent and parameter extraction                  |
| **Storage**             | MinIO                                | S3-compatible artifact storage                   |
| **Environment Manager** | uv                                   | Modern dependency + environment management       |

---

## 🧠 **Core Workflow**

1. **Receive Request**
   Guru receives an A2A message containing:

   * Dataset (file or URL)
   * Natural language instruction

2. **Parse Instruction**
   spaCy extracts task intent, numeric references, and column mentions.

3. **Interpret Query (LLM)**
   Qwen2.5-7B converts the instruction into a **strictly-defined JSON schema** understood by Guru.

4. **Perform Analysis**
   pandas loads the dataset and executes the requested operation.

5. **Visualize Results**
   matplotlib generates relevant charts.

6. **Store Artifacts**
   Binary chart outputs are uploaded to MinIO; URLs are returned.

7. **Respond to Telex**
   Guru returns structured JSON compatible with Telex A2A message formats.

---

## 🧰 **Setup and Installation**

### **1️⃣ Clone the Repository**

```
git clone https://github.com/<your-username>/guru-data-insight-agent.git
cd guru-data-insight-agent
```

### **2️⃣ Install Dependencies**

Using **uv** (recommended):

```
uv sync
```

Or with **pip**:

```
pip install -r requirements.txt
```

### **3️⃣ Create Environment Variables**

Create a `.env` file:

```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET_NAME=guru-artifacts
```

### **4️⃣ Run the Server**

```
uv run uvicorn data_insight_agent.main:app --reload
```

Guru will be available at:

```
http://127.0.0.1:8000 (Test with PostMan)
```

---

## 🧠 **Example Flow**

### **Input (A2A Message)**

```
{
  "instruction": "Show a regression between age and income",
  "dataset_url": "https://example.com/people.csv"
}
```

### **Output (Response to Telex)**

```
{
  "message": "Regression between age and income completed successfully.",
  "artifact": {
    "type": "image",
    "url": "https://minio.example.com/guru-artifacts/abc123.png"
  },
  "summary": "Income increases linearly with age up to mid-40s."
}
```

---

## 📊 **Visualization Types Supported**

* **bar** — category comparison
* **line** — trends over time
* **scatter** — correlations
* **hist** — distributions

Guru uses a **clean dictionary-based visualization dispatch**, avoiding repetitive `if/else` blocks.

---

## 🧱 **Development Notes**

* Developed on **Linux (WSL)**
* Managed using **uv** for clean environment isolation
* `pipx` used for external tools (Jupyter, uv)
* All artifacts uploaded to MinIO as **binary streams**
* Strict schema enforcement for LLM responses
* Modularized for future expansion and plug-in operations

---

## 🔮 **Future Improvements**

* 🔗 Automated A2A registration & handshake
* 🧠 More advanced NLP models for richer parsing
* 📊 Support for multi-dataset comparative analysis
* 📈 Optional integration with Plotly/Seaborn for enhanced visuals
* 📝 Insight summarization via hybrid rule-based + LLM reasoning

---

## 👨‍💻 **Author**

**Faruq Alabi Bashir**
Backend Engineer • Data Insight Developer

GitHub: [https://github.com/MR-RITZY](https://github.com/MR-RITZY)
Email: [faruqbashir608@gmail.com](mailto:faruqbashir608@gmail.com)
