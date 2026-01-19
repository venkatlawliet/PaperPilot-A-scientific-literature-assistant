-- Enable UUID extension (required for uuid_generate_v4)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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
