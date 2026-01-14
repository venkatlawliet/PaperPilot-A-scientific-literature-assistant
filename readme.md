# 🚀 PaperPilot: A Scientific Literature Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![Deployed](https://img.shields.io/badge/deployed-AWS%20EC2-FF9900.svg)](https://aws.amazon.com/ec2/)

> **A production-grade AI system that finds research papers from the internet and answers questions grounded strictly in their content — with multi-tenant architecture.**

---

## 📋 Table of Contents

- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Solution Overview](#-solution-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Performance & Evaluation](#-performance--evaluation)
- [Technical Deep Dive](#-technical-deep-dive)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [Deployment](#-deployment)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Motivation & Problem Statement

### The Real-World Challenge

As a research assistant, I faced a recurring problem: **reading entire research papers to answer specific questions was extremely time-consuming**. A typical workflow looked like this:

- 📄 **Find the paper in the internet by go through multiple websites** → 1 minutes
- 📖 **Skim abstract and introduction** → 5 minutes
- 🔍 **Find relevant sections based on the question i am having in my mind** → 2 minutes
- 📝 **read specific information and get answers(sometimes i had to read multiple sections to check if that was the only answer)** → 3-5 minutes
- ⏱️ **Total time per question: 10+ minutes**

This became hard to sustain when juggling multiple papers or tackling complex questions. It takes up most of my time and leaves very little room to actually run experiments—especially since, as researchers, we often get unexpected results about 70% of the time.

### Why Existing Solutions Fall Short

| Approach | Limitation |
|----------|-----------|
| **General LLMs (ChatGPT, Claude)** | Summarize papers, but don't have access to the actual document during inference → potential hallucinations and lack of grounding |
| **Simple RAG systems** | Use only dense embeddings → poor performance on keyword-specific queries (e.g., "What was the accuracy on ImageNet?") |
| **Manual reading** | Accurate and the best approach but extremely slow |

### What I Needed

A system that could:
1. **Fetch any research paper** from the internet automatically
2. **Answer questions grounded strictly in the paper's content** (no hallucinations)
3. **Respond in under limited time** (as i dont wanna spend 10+ mins manually)
4. **Handle both explanatory and numerical queries** with high accuracy
5. **Generate diagrams** from paper content when requested
6. **Work across all research domains** without domain-specific configuration

---

## 💡 Solution Overview

**PaperPilot** is a production-grade Retrieval-Augmented Generation (RAG) system designed specifically for research paper analysis. It combines:

- ✅ **Hybrid search** (dense + sparse embeddings) for 30% better retrieval quality
- ✅ **Intelligent orchestration** via Claude for automatic tool selection
- ✅ **Diagram generation** using LLMs + D2 language
- ✅ **Multi-tenant architecture** with complete user isolation
- ✅ **Production deployment** on AWS EC2 with CI/CD

### Design Philosophy

1. **Accuracy First**: Traditional RAG with careful retrieval control over experimental approaches (e.g., knowledge graphs)
2. **Domain Agnostic**: No hardcoded schemas — works with any research paper
3. **Grounded Responses**: All answers cite page numbers and are strictly derived from document context
4. **Production Ready**: Full containerization, testing, and automated deployment

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit Web App<br/>Dark Theme, Multi-Modal Chat]
    end
    
    subgraph "Authentication & Multi-Tenancy"
        AUTH[User Auth<br/>bcrypt hashing]
        NS[Namespace Isolation<br/>user_username]
    end
    
    subgraph "Orchestration Layer"
        CLAUDE[Claude API<br/>Intent Classification]
        CLAUDE -->|direct answer| KNOWLEDGE[Claude's Knowledge]
        CLAUDE -->|web_search| WEB[SerpAPI + Llama]
        CLAUDE -->|research_lookup| RESEARCH[Research Mode]
    end
    
    subgraph "Paper Discovery & Ingestion"
        S2[Semantic Scholar API<br/>Paper Search]
        RESOLVE[Content Resolver<br/>PDF URL Extraction]
        ADE[LandingAI ADE<br/>dpt-2-latest Model]
        S2 --> RESOLVE
        RESOLVE --> ADE
    end
    
    subgraph "Embedding Pipeline"
        ADE --> CHUNK[Text Chunking<br/>with Page Metadata]
        CHUNK --> DENSE[Dense Embeddings<br/>all-mpnet-base-v2<br/>768 dimensions]
        CHUNK --> SPARSE[Sparse Embeddings<br/>BM25 Encoder<br/>Keyword-based]
        DENSE --> HYBRID[Hybrid Vectors<br/>α=0.6 dense + 0.4 sparse]
        SPARSE --> HYBRID
    end
    
    subgraph "Storage Layer"
        PINECONE[(Pinecone<br/>Vector Database<br/>Multi-tenant namespaces)]
        SUPABASE[(Supabase PostgreSQL)]
        
        SUPABASE -.->|tables| USERS[users]
        SUPABASE -.->|tables| PAPERS[papers + bm25_state]
        SUPABASE -.->|tables| CHUNKS[paper_chunks]
        SUPABASE -.->|tables| CHATS[paper_chats]
    end
    
    subgraph "Query Processing"
        QREWRITE[Groq Llama<br/>Query Rewriting<br/>for Diagrams]
        BM25REFIT[BM25 Refitting<br/>from stored chunks]
        HSEARCH[Hybrid Search<br/>Top-5 Retrieval<br/>namespace + paper_id filter]
        QREWRITE --> BM25REFIT
        BM25REFIT --> HSEARCH
    end
    
    subgraph "Answer Generation"
        LLMBRIDGE[LLM Bridge<br/>Claude/Llama]
        D2GEN[D2 Code Generation<br/>Llama + Context]
        D2CLI[D2 CLI<br/>SVG Rendering]
        HSEARCH --> LLMBRIDGE
        HSEARCH --> D2GEN
        D2GEN --> D2CLI
    end
    
    UI --> AUTH
    AUTH --> CLAUDE
    CLAUDE --> RESEARCH
    RESEARCH --> S2
    HYBRID --> PINECONE
    CHUNK --> SUPABASE
    
    UI -.->|load past paper| SUPABASE
    SUPABASE -.->|chunks| BM25REFIT
    PINECONE --> HSEARCH
    
    LLMBRIDGE --> UI
    D2CLI --> UI
    
    style CLAUDE fill:#2563eb,stroke:#1e40af,color:#fff
    style PINECONE fill:#6366f1,stroke:#4f46e5,color:#fff
    style SUPABASE fill:#10b981,stroke:#059669,color:#fff
    style ADE fill:#f59e0b,stroke:#d97706,color:#fff
    style UI fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

### Flow Explanation

1. **User Authentication** → Multi-tenant namespace assignment
2. **Query Intent Classification** → Claude decides: knowledge / web search / research mode
3. **Paper Discovery** → Semantic Scholar → Human-in-the-loop confirmation
4. **PDF Extraction** → LandingAI ADE converts everything to structured text
5. **Hybrid Embedding** → Dense (semantic) + Sparse (keyword) vectors
6. **Dual Storage**:
   - Pinecone: Hybrid vectors with metadata
   - Supabase: Full text chunks + BM25 state for refitting
7. **Query Processing**:
   - Diagram requests → Groq rewrites query for better retrieval
   - BM25 refitted on stored chunks for consistent sparse encoding
   - Hybrid search retrieves top-5 relevant chunks
8. **Answer Generation**:
   - Regular Q&A → Claude/Llama with strict context adherence
   - Diagrams → Llama generates D2 code → D2 CLI renders SVG

---

## ⚡ Key Features

### 1. Intelligent Query Routing

```python
# Claude automatically decides the best approach
User: "What is the capital of France?"
→ Direct answer from Claude's knowledge

User: "What's the current price of Bitcoin?"
→ SerpAPI web search + Llama synthesizes answer

User: "Find the paper on Vision Transformers"
→ Activates research mode → Semantic Scholar search
```

### 2. Hybrid Search (Dense + Sparse)

Traditional RAG systems use only dense embeddings, which struggle with:
- Exact keyword matches
- Numerical queries ("What was the accuracy?")
- Named entities ("Who is the first author?")

PaperPilot combines:
- **Dense embeddings** (`all-mpnet-base-v2`): Semantic understanding
- **Sparse embeddings** (BM25): Keyword matching

**Result: 30% improvement in retrieval quality**

### 3. Universal Text Representation

Unlike systems that maintain separate indexes for text, tables, and images, PaperPilot uses **LandingAI's Document Pre-trained Transformer (DPT-2)** to extract everything as structured text:

- ✅ Tables → Markdown tables
- ✅ Equations → LaTeX/text
- ✅ Figures → Descriptive text
- ✅ Headers → Structured sections

**Benefit**: Single unified index, reduced latency, simpler architecture

### 4. Diagram Generation

When users request visualizations:

```
User: "Generate an architecture diagram from this paper"

System:
1. Rewrites query → "What is the architecture described in the paper?"
2. Retrieves relevant chunks via hybrid search
3. Llama interprets context → Writes D2 diagram code
4. D2 CLI renders → SVG image displayed in UI
```

Example D2 code generated:
```d2
Input Layer -> Feature Extractor: raw data
Feature Extractor -> Encoder: feature maps
Encoder -> Latent Space: compressed representation
Latent Space -> Decoder: reconstruction
Decoder -> Output Layer: final result
```

### 5. Multi-Tenant Architecture

Complete isolation between users:
- Each user gets unique **Pinecone namespace** (`user_username`)
- All queries filter by `namespace + paper_id`
- Hashed passwords with bcrypt
- PostgreSQL for user data and paper metadata

### 6. BM25 State Persistence

**The Challenge**: BM25 must be fitted on the same corpus for consistent sparse embeddings.

**The Solution**:
1. Store all chunk texts in Supabase `paper_chunks` table
2. When user loads a past paper → Refit BM25 on stored texts
3. Fallback: Store BM25 state as JSON in `papers.bm25_state` column

### 7. Production Infrastructure

- 🐳 **Docker**: Multi-stage builds, pre-downloaded models
- ☁️ **AWS EC2**: t3.small instance, production deployment
- 🔄 **CI/CD**: GitHub Actions (lint → test → build → push to Docker Hub)
- 🧪 **Testing**: Pytest suite with 83% coverage
- 📊 **Monitoring**: Structured logging throughout

---

## 📊 Performance & Evaluation

### Speed Improvement

| Metric | Manual Process | PaperPilot | Improvement |
|--------|---------------|------------|-------------|
| Regular Q&A | 10-20+ minutes | 1-1.5 seconds | **400-1200x faster** |
| Diagram generation | N/A (manual drawing) | 2-3 seconds | Fully automated |
| Paper ingestion | N/A | 15-30 seconds | One-time cost |

### Retrieval Quality Evaluation

Tested on **80 evaluation queries** (40 explanatory + 40 numerical/integer-based):

| Retrieval Method | Explanatory Q&A | Numerical Q&A | Average |
|-----------------|----------------|---------------|---------|
| **Dense-only (baseline)** | 28/40 (70%) | 20/40 (50%) | 60% |
| **Hybrid (dense + sparse)** | 37/40 (92.5%) | 30/40 (75%) | 83.75% |
| **Improvement** | +32% | +50% | **+39.5%** |

**Key Insight**: Sparse embeddings (BM25) dramatically improve performance on keyword-specific and numerical queries.

### QA Correctness Evaluation

Evaluated on **40 fixed test queries** with ground-truth answers:

- ✅ **Correct answers**: 33/40
- ❌ **Incorrect answers**: 7/40
- **Accuracy**: **82.5% ≈ 83%**

**Error Analysis**:
- 4 cases: Context retrieved but LLM misinterpreted
- 2 cases: Relevant chunk ranked 6-10 (outside top-5)
- 1 case: Complex multi-hop reasoning required

---

## 🔬 Technical Deep Dive

### Why Not Knowledge Graphs?

I initially explored knowledge graph-based approaches to improve grounding accuracy. However, I encountered fundamental limitations:

#### Challenges with KG Approach:

1. **Schema Definition Problem**:
   - KGs require well-defined schemas (entities, relationships)
   - Works well for domain-specific systems (e.g., medical, legal)
   - **Fails for general research papers** across diverse domains (CS, philosophy, biology, physics)
   - No universal schema that generalizes well

2. **Hallucination Risk**:
   - Using LLMs to construct graphs can introduce hallucinated entities/relationships
   - Difficult to validate if extracted relationships faithfully represent the source paper
   - Risk of misleading downstream reasoning

3. **Complexity vs. Reliability Trade-off**:
   - KGs add significant complexity (schema design, entity extraction, relation extraction, graph storage)
   - I believe if you wanted to build a good system, **reliability > sophistication** | **reliability > capabilities**
   - RAG with careful retrieval control proved more robust

**In long term**: Focus on robust hybrid RAG system now. Revisit KG integration in future with better validation mechanisms.

### Multi-Tenant Isolation Strategy

```python
# Each user gets unique namespace
def _safe_namespace(username: str) -> str:
    safe = username.lower().strip().replace(" ", "_")
    safe = "".join(c for c in safe if (c.isalnum() or c == "_"))
    return f"user_{safe}"

# Example:
# username: "Alice Smith" → namespace: "user_alice_smith"
# username: "bob@research" → namespace: "user_bobresearch"

# All Pinecone queries filtered by namespace + paper_id
index.query(
    vector=dense_query,
    sparse_vector=sparse_query,
    namespace=user.namespace,
    filter={"paper_id": {"$eq": float(paper_id)}}
)
```

**Benefits**:
- Complete data isolation between users
- Each user can have multiple papers
- No cross-contamination in search results
- Scalable to millions of users

### Hybrid Search Implementation

```python
def query_ade_index(
    query: str,
    bm25: BM25Encoder,
    dense_model,
    namespace: str,
    paper_id: int,
    top_k: int = 5,
    alpha: float = 0.6,  # Weight for dense vs sparse
):
    # Dense vector (semantic)
    q_dense = dense_model.encode([query])[0].tolist()
    
    # Sparse vector (keyword)
    q_sparse = bm25.encode_queries([query])[0]
    
    # Alpha weighting: 0.6 dense + 0.4 sparse
    weighted_sparse = {
        "indices": q_sparse["indices"],
        "values": [v * (1 - alpha) for v in q_sparse["values"]]
    }
    weighted_dense = [v * alpha for v in q_dense]
    
    # Pinecone hybrid query
    return index.query(
        vector=weighted_dense,
        sparse_vector=weighted_sparse,
        top_k=top_k,
        namespace=namespace,
        filter={"paper_id": {"$eq": float(paper_id)}}
    )
```

**Why Alpha = 0.6?**
- Empirically tested on validation set
- 0.6 dense weights semantic understanding more heavily
- 0.4 sparse ensures keyword matches aren't ignored
- Balances recall and precision

### Query Rewriting for Diagrams

Challenge: "Generate architecture diagram" doesn't appear literally in papers.

Solution: Groq Llama rewrites diagram requests into factual queries:

```python
# User query
"Generate a diagram of the data flow in the paper"

# Rewritten query (for retrieval)
"What is the data flow described in the paper?"

# After retrieval, pass ORIGINAL query + context to diagram generator
# This preserves user intent while improving chunk retrieval
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- API Keys for:
  - [Anthropic Claude](https://console.anthropic.com/)
  - [Groq](https://console.groq.com/)
  - [Pinecone](https://www.pinecone.io/)
  - [Supabase](https://supabase.com/)
  - [Semantic Scholar](https://www.semanticscholar.org/product/api)
  - [LandingAI](https://landing.ai/)
  - [SerpAPI](https://serpapi.com/)

### Installation

#### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/venkatlawliet/PaperPilot-A-scientific-literature-assistant.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install D2 CLI for diagram generation
curl -fsSL https://d2lang.com/install.sh | sh -s --

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

#### Option 2: Docker Setup

```bash
# Pull from Docker Hub
docker pull (Image link will be shared on request)

# Or build locally
docker build -t paperpilot .

# Run container
docker run -p 8501:8501 \
  --env-file .env \
  (Image link will be shared on request)
```

### Configuration

Edit `.env` file with your credentials:

```env
# Supabase (Database)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# Pinecone (Vector Database)
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=researchmcp

# LLM APIs
CLAUDE_API_KEY=your-claude-key
GROQ_API_KEY=your-groq-key

# Paper Search & Extraction
S2_API_KEY=your-semantic-scholar-key
VISION_AGENT_API_KEY=your-landingai-key

# Web Search
SERPAPI_API_KEY=your-serpapi-key

# Optional
UNPAYWALL_EMAIL=you@example.com  # For open access papers
```

### Database Setup

1. Create Supabase project at [supabase.com](https://supabase.com/)

2. Run the following SQL to create tables:

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    namespace TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Papers table
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    bm25_state JSONB,
    status TEXT DEFAULT 'ingested',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Paper chunks table (for BM25 refitting)
CREATE TABLE paper_chunks (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    page INTEGER,
    type TEXT,
    caption TEXT,
    text TEXT NOT NULL,
    grounding INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat history table
CREATE TABLE paper_chats (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    d2_code TEXT,
    svg_path TEXT,
    source_type TEXT DEFAULT 'paper',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_papers_user_id ON papers(user_id);
CREATE INDEX idx_chunks_paper_id ON paper_chunks(paper_id);
CREATE INDEX idx_chats_paper_id ON paper_chats(paper_id);
```

3. Create Pinecone index:
   - Dimension: 768
   - Metric: dotproduct
   - Cloud: AWS
   - Region: us-east-1

### Running the Application

```bash
# Start Streamlit app
streamlit run frontend.py

# Access at http://localhost:8501
```

---

## 📖 Usage Guide

### 1. Registration & Login

- Navigate to `http://localhost:8501`
- Create an account (username + password)
- Login with credentials

### 2. General Chat Mode

Ask anything without loading a paper:

```
You: "What is machine learning?"
→ Claude answers from its knowledge

You: "What's the current weather in New York?"
→ System uses SerpAPI + Llama to answer

You: "Find papers about BERT"
→ System activates research mode
```

### 3. Research Mode - Finding Papers

**Option A: Search Semantic Scholar**

1. System activates research lookup when you mention papers
2. Enter paper title or keywords
3. System shows top 5 results from Semantic Scholar
4. Select paper to ingest
5. System extracts PDF content (15-30 seconds)

**Option B: Upload PDF**

1. Click "Upload Paper" tab
2. Select PDF file from your computer
3. Enter paper title
4. System ingests immediately

### 4. Querying Papers

Once a paper is ingested:

```
You: "What was the main contribution?"
→ System: [retrieves context] → [Claude answers with page citations]

You: "What accuracy did they achieve on ImageNet?"
→ System: [hybrid search finds numerical mentions] → [answers with page]

You: "Show me the architecture diagram"
→ System: [rewrites query] → [retrieves chunks] → [Llama generates D2 code] → [renders SVG]
```

### 5. Loading Past Papers

- Click "Your Ingested Papers" dropdown
- Select previously ingested paper
- System loads chat history and refits BM25
- Continue asking questions

### 6. Switching Modes

- In research mode: Click "Exit Paper Mode" → Returns to general chat
- General chat mode: Mention a paper → Activates research lookup

---

## 🌐 Deployment

### AWS EC2 Deployment

1. **Launch EC2 Instance**:
   ```bash
   # Instance type: t3.small (or larger)
   # AMI: Ubuntu 22.04 LTS
   # Security group: Allow inbound port 8501
   ```

2. **Install Docker on EC2**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker ubuntu
   ```

3. **Pull and Run Container**:
   ```bash
   # Create .env file on EC2
   nano .env  # Paste your API keys
   
   # Pull image
   (Image link can be shared upon request)
   
   # Run container
   docker run -d -p 8501:8501 \
     --name paperpilot \
     --restart unless-stopped \
     --env-file .env \
     (Image link can be shared upon request)
   ```

4. **Access Application**:
   ```
   http://<EC2-PUBLIC-IP>:8501
   ```

### CI/CD Pipeline

GitHub Actions automatically:
1. ✅ Lints code (Black, isort, flake8)
2. ✅ Runs pytest suite
3. ✅ Builds Docker image
4. ✅ Pushes to Docker Hub on merge to main

Workflow file: `.github/workflows/ci-cd.yml`

To enable:
1. Add Docker Hub credentials as GitHub Secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
2. Push to main branch → Automated deployment

---

## 🔮 Future Work

### Short-term Improvements

- [ ] **Extended Context Windows**: Integrate more intelligent language model for token context with cheap cost
- [ ] **Batch Ingestion**: Process multiple papers simultaneously
- [ ] **Advanced Caching**: Redis layer for frequently accessed chunks
- [ ] **Citation Extraction**: Automatically extract and link paper citations
- [ ] **Collaborative Features**: Share papers and annotations between users

### Medium-term Enhancements

- [ ] **Agentic Workflows**: Multi-step reasoning across multiple papers
- [ ] **Voice Interface**: Speech-to-text for hands-free paper analysis
- [ ] **Mobile App**: Native iOS/Android applications
- [ ] **Export Features**: Generate PDF reports with answers and diagrams
- [ ] **Comparison Mode**: Compare methods/results across multiple papers and produce camparision visualizations(graphs/tables)

### Long-term Research Directions

- [ ] **Knowledge Graph Integration**: Hybrid KG + RAG with validation mechanisms
- [ ] Although i first need to need research and come up with a way to predefine a schema that works for non domain specific workflows(This is a very good research but takes you months and years to achieve and yes, I’m a victim of this process — but I still haven’t lost interest.)
  - Entity extraction with confidence scoring
  - Relationship verification against source text
  - Schema learning from paper corpus
  - Use KG for complex multi-hop reasoning
  
- [ ] **Active Learning**: Learn from user corrections to improve retrieval
- [ ] **Multimodal Understanding**: Process images, tables, equations natively
- [ ] **Domain Adaptation**: Fine-tune models per research domain
- [ ] **Automated Literature Review**: Generate comprehensive reviews from paper corpus

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork repository
git clone https://github.com/YOUR_USERNAME/PaperPilot-A-scientific-literature-assistant.git

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt
pip install black isort flake8 pytest pytest-cov

# Make changes and test
pytest tests/ -v
black .
isort .
flake8 .

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

### Pull Request Process

1. Ensure all tests pass (`pytest tests/`)
2. Add tests for new features
3. Update README if adding major features
4. Follow conventional commit messages (`feat:`, `fix:`, `docs:`, etc.)
5. Request review from maintainers

### Code Style

- **Formatter**: Black with line length 88
- **Import sorting**: isort
- **Linter**: flake8
- **Type hints**: Encouraged but not required

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Technologies Used

- **LLMs**: [Anthropic Claude](https://www.anthropic.com/), [Groq Llama](https://groq.com/)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **Database**: [Supabase](https://supabase.com/)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/)
- **PDF Extraction**: [LandingAI ADE](https://landing.ai/)
- **Paper Search**: [Semantic Scholar API](https://www.semanticscholar.org/)
- **Web Search**: [SerpAPI](https://serpapi.com/)
- **Diagram Generation**: [D2 Language](https://d2lang.com/)
- **UI Framework**: [Streamlit](https://streamlit.io/)

### Inspiration

This project was born out of a real need during my work as a research assistant. The frustration of spending 10+ minutes per question led to building a system that now answers in under 2 seconds while maintaining 83% accuracy. I might have spent months building this project but i learnt a lot of things during my journey. 

### Research Papers

Key papers that influenced this work:
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
- *Dense Passage Retrieval for Open-Domain Question Answering* (Karpukhin et al., 2020)
- *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (Thakur et al., 2021)
- *Precise Zero-Shot Dense Retrieval without Relevance Labels* (Gao et al., 2022)

---

## 📧 Contact

**Venkat**  
- GitHub: [@venkatlawliet](https://github.com/venkatlawliet)
- Email: venkatoffice6o6@gmail.com

---

## ⭐ Star History

If you find this project useful, please consider giving it a star:) ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=venkatlawliet/PaperPilot-A-scientific-literature-assistant&type=Date)](https://star-history.com/#venkatlawliet/PaperPilot-A-scientific-literature-assistant&Date)

---

<div align="center">
  <sub>Built with ❤️ by researchers, for researchers</sub>
</div>
