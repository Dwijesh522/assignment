Here is the complete README in proper `.md` format. You can copy-paste it directly into your `README.md` file.

````markdown
# Mini RAG System

## Setup

Create virtual environment and install dependencies:

```bash
python3 -m venv rag_env
source rag_env/bin/activate
pip3 install -r requirements.txt
````

Set OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## Build Index

Run the following command to process emails and build the FAISS index:

```bash
python3 build_index.py --emails_dir emails --out_dir faiss_index
```

---

## Run Queries

Start the query interface:

```bash
python3 run_query.py --index_dir faiss_index
```

Type your query and press Enter.

To exit:

```bash
exit
```

```
```

