import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

model = SentenceTransformer('all-MiniLM-L6-v2')


client = QdrantClient("localhost", port=6333)

client.recreate_collection(
    collection_name="js_chunks",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)


with open("input.html", "r", encoding="utf-8") as f:
    html_content = f.read()


soup = BeautifulSoup(html_content, "lxml")


js_blocks = [script.get_text() for script in soup.find_all("script")]


js_code = "\n".join(js_blocks)

js_patterns = {
    "function_declaration": r'(function\s+\w+\s*\([^)]*\)\s*\{[^}]*\})',
    "async_function": r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{[^}]*\})',
    "event_listener": r'(addEventListener\s*\(\s*[\'"]\w+[\'"]\s*,\s*function\s*\([^)]*\)\s*\{[^}]*\}\))',
    "arrow_function": r'(const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*\})'
}

js_chunks = []
for label, pattern in js_patterns.items():
    matches = re.findall(pattern, js_code, flags=re.DOTALL)
    for match in matches:
        js_chunks.append({
            "type": label,
            "content": match.strip()
        })


id_counter = 0
for chunk in js_chunks:
    embedding = model.encode(chunk['content']).tolist()
    client.upsert(
        collection_name="js_chunks",
        points=[
            models.PointStruct(
                id=id_counter,
                vector=embedding,
                payload=chunk
            )
        ]
    )
    id_counter += 1

print(f"Inserted {id_counter} JS chunks into vector DB")
