import tinycss2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

# --- Setup embedding model ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Initialize Qdrant client ---
qdrant_client = QdrantClient("localhost", port=6333)

# --- (Re)create Qdrant collection ---
qdrant_client.recreate_collection(
    collection_name="css_chunks",
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    }
)

# --- Initialize LangChain VectorStore ---
vector_store = Qdrant(
    client=qdrant_client,
    collection_name="css_chunks",
    embedding=embedding_model
)

# --- Read HTML file ---
with open("input.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# --- Extract all <style> tags' CSS content ---
soup = BeautifulSoup(html_content, "html.parser")
css_content = "\n".join(style.get_text() for style in soup.find_all('style'))

# --- Parse CSS rules using tinycss2 ---
rules = tinycss2.parse_stylesheet(css_content)

# --- Prepare LangChain Documents ---
documents = []
id_counter = 0

for rule in rules:
    if rule.type == 'qualified-rule':
        selector = tinycss2.serialize(rule.prelude).strip()
        properties = tinycss2.serialize(rule.content).strip()
        full_rule = f"{selector} {{ {properties} }}"

        metadata = {
            "_id": id_counter,
            "type": "css_rule",
            "selector": selector,
            "properties": properties,
            "full_rule": full_rule
        }

    elif rule.type == 'at-rule' and rule.lower_at_keyword == 'keyframes':
        name = tinycss2.serialize(rule.prelude).strip()
        content = tinycss2.serialize(rule.content).strip()
        full_rule = f"@keyframes {name} {{ {content} }}"

        metadata = {
            "_id": id_counter,
            "type": "keyframes_rule",
            "name": name,
            "content": content,
            "full_rule": full_rule
        }

    elif rule.type == 'at-rule' and rule.lower_at_keyword == 'media':
        condition = tinycss2.serialize(rule.prelude).strip()
        content = tinycss2.serialize(rule.content).strip()
        full_rule = f"@media {condition} {{ {content} }}"

        metadata = {
            "_id": id_counter,
            "type": "media_rule",
            "condition": condition,
            "content": content,
            "full_rule": full_rule
        }

    else:
        continue

    doc = Document(
        page_content=full_rule,
        metadata=metadata
    )

    documents.append(doc)
    print(f"Prepared doc {id_counter}: {full_rule}")
    id_counter += 1

# --- Upload documents into Vector DB ---
vector_store.add_documents(documents)

print(f"✅ Total {len(documents)} CSS chunks inserted into vector DB via LangChain.")
