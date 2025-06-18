import tinycss2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from bs4 import BeautifulSoup

# Initialize embedding model and Qdrant client
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient("localhost", port=6333)

# Recreate collection for CSS chunks
client.recreate_collection(
    collection_name="css_chunks",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

# Read CSS file
with open("input.html", "r", encoding="utf-8") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, "html.parser")
css_content = "\n".join(style.get_text() for style in soup.find_all('style'))

# Parse CSS stylesheet into rules
rules = tinycss2.parse_stylesheet(css_content)

id_counter = 0

# Process each rule block
for rule in rules:
    if rule.type == 'qualified-rule':
        selector = tinycss2.serialize(rule.prelude).strip()
        properties = tinycss2.serialize(rule.content).strip()
        full_rule = f"{selector} {{ {properties} }}"

        chunk = {
            "type": "css_rule",
            "name": selector,
            "content": properties,
            "page_content": full_rule
        }

    elif rule.type == 'at-rule' and rule.lower_at_keyword == 'keyframes':
        name = tinycss2.serialize(rule.prelude).strip()
        content = tinycss2.serialize(rule.content).strip()
        full_rule = f"@keyframes {name} {{ {content} }}"

        chunk = {
            "type": "keyframes_rule",
            "name": name,
            "content": content,
            "page_content": full_rule
        }
    elif rule.type == 'at-rule' and rule.lower_at_keyword == 'media':
            condition = tinycss2.serialize(rule.prelude).strip()
            content = tinycss2.serialize(rule.content).strip()
            full_rule = f"@media {condition} {{ {content} }}"

            chunk = {
                "type": "media_rule",
                "name": condition,
                "content": content,
                "page_content": full_rule
            }

    else:
        continue 

    # Vectorize and insert
    embedding = model.encode(chunk['page_content']).tolist()
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
    print(f"Inserted chunk {id_counter}: {chunk['page_content']}")
    id_counter += 1

print(f"✅ Total {id_counter} CSS chunks inserted into vector DB.")
