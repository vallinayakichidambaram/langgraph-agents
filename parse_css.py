import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


model = SentenceTransformer('all-MiniLM-L6-v2')


client = QdrantClient("localhost", port=6333)


client.recreate_collection(
    collection_name="css_chunks",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)


with open("input.html", "r", encoding="utf-8") as f:
    html_content = f.read()


soup = BeautifulSoup(html_content, "lxml")


css_blocks = [style.get_text() for style in soup.find_all('style')]
css_patterns = {
    "css_variable_block": r'(:root)\s*\{([^}]+)\}',
    "global_selector": r'((?:body|\*|html|[a-zA-Z0-9\-]+))\s*\{([^}]+)\}',
    "class_selector": r'(\.[\w\-]+)\s*\{([^}]+)\}',
    "keyframes": r'(@keyframes\s+[\w\-]+)\s*\{([^}]+)\}'
}


css_chunks = []
for css in css_blocks:
    for label, pattern in css_patterns.items():
        matches = re.findall(pattern, css)
        for match in matches:
            selector_or_name, rules = match
            content = f"{selector_or_name} {{{rules.strip()}}}"
            css_chunks.append({
                "type": label,
                "name": selector_or_name,
                "content": content
            })
id_counter = 0

for chunk in css_chunks:
    embedding = model.encode(chunk['content']).tolist()
    client.upsert(
        collection_name="css_chunks",
        points=[
            models.PointStruct(
                id=id_counter,
                vector=embedding,
                payload=chunk
            )
        ]
    )
    id_counter += 1

print(f"Inserted {id_counter} chunks into vector DB")