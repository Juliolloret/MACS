# Agent Memory

## Introduction

Agents are systems that intelligently accomplish tasks, ranging from executing simple workflows to pursuing complex, open-ended objectives. To be effective, agents need to learn from their interactions and have access to knowledge that extends beyond their initial training data. This is where memory comes in.

Knowledge and memory help agents store, retrieve, and utilize information. This capability is crucial for building robust agentic systems that can handle a wide variety of tasks. This guide explores how to build agents with long-term memory using OpenAI's primitives.

## Core Concepts: Embeddings and Vector Stores

The foundation of agent memory lies in two key concepts: **embeddings** and **vector stores**.

*   **Embeddings** are numerical representations of text (or other data types). They are designed such that semantically similar pieces of text have embeddings that are close to each other in a high-dimensional space. This allows for a nuanced understanding of meaning beyond simple keywords.
*   **Vector Stores** are databases designed to store and index these embeddings for efficient retrieval. When an agent needs to recall information, it can query the vector store with a piece of text. The vector store then finds the most relevant stored information based on the semantic similarity of their embeddings.

Together, embeddings and vector stores provide a powerful mechanism for an agent to have a long-term memory that it can search through to inform its actions.

## Embeddings

OpenAI’s text embeddings measure the relatedness of text strings. An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

Embeddings are commonly used for:

*   **Search** (where results are ranked by relevance to a query string)
*   **Clustering** (where text strings are grouped by similarity)
*   **Recommendations** (where items with related text strings are recommended)
*   **Classification** (where text strings are classified by their most similar label)

### How to Get Embeddings

To get an embedding, you send your text string to the embeddings API endpoint along with an embedding model name.

**Python Example:**
```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)

print(response.data[0].embedding)
```

### Embedding Models

OpenAI offers several embedding models, each with different performance characteristics and costs.

|Model|~ Pages per dollar|Performance on MTEB eval|Max input|
|---|---|---|---|
|text-embedding-3-small|62,500|62.3%|8192|
|text-embedding-3-large|9,615|64.6%|8192|
|text-embedding-ada-002|12,500|61.0%|8192|

## Vector Stores for Retrieval

Vector stores are the containers that power semantic search. When you add a file to a vector store it will be automatically chunked, embedded, and indexed for you.

The Retrieval API allows you to perform semantic search over your data, which is a technique that surfaces semantically similar results — even when they match few or no keywords.

### Performing Semantic Search

You can query a vector store using the `search` function and specifying a `query` in natural language. This will return a list of relevant chunks from your documents.

**Python Example:**
```python
from openai import OpenAI
client = OpenAI()

# Assume vector_store_id is created and has files
vector_store_id = "vs_123"

user_query = "What is the return policy?"

results = client.vector_stores.search(
    vector_store_id=vector_store_id,
    query=user_query,
)

print(results)
```

## Long-term Storage in Files

While the Retrieval API and vector stores offer a managed solution, you might want more control over your embeddings, or to manage them locally. A common pattern is to generate embeddings for your documents and store them in local files, like a CSV. This gives you a persistent, long-term storage of your knowledge base that you can then load into any vector database or use for local similarity searches.

Here's an example of how you can generate embeddings for a dataset and save them to a CSV file using pandas.

```python
import pandas as pd
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Assuming you have a pandas DataFrame `df` with a 'text' column
# For example:
data = {'text': ['I have bought several of the Vitality canned dog food products and have found them all to be of good quality.',
                 'Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor intended to represent the product as "Jumbo".']}
df = pd.DataFrame(data)


df['embedding'] = df.text.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('embedded_reviews.csv', index=False)
```

You can then load these pre-computed embeddings from the file for use in your application, avoiding the need to re-compute them every time.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('embedded_reviews.csv')
# The embedding is stored as a string, so we need to convert it back to a list/array
df['embedding'] = df.embedding.apply(eval).apply(np.array)
```

This approach provides a simple yet powerful way to manage long-term memory for your agent using plain files.

## Synthesizing Responses

After retrieving relevant information from memory, the next step is often to synthesize a response. You can use a powerful language model like GPT-4o to generate a coherent answer based on the retrieved sources and the original user query.

Here's how you can combine retrieval results with a chat model to generate a grounded response:

```python
# Continuing from the vector store search example...
# user_query = "What is the return policy?"
# results = client.vector_stores.search(...)

# Format the results into a string to pass to the model
# (Note: a robust implementation would handle multiple results and content parts)
formatted_results = ""
if results.data:
    first_result = results.data[0]
    if first_result.content:
        formatted_results = first_result.content[0].text

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Produce a concise answer to the query based on the provided sources."
        },
        {
            "role": "user",
            "content": f"Sources: {formatted_results}\\n\\nQuery: '{user_query}'"
        }
    ],
)

print(completion.choices[0].message.content)
```

This pattern of retrieve-then-synthesize is a cornerstone of building knowledgeable and reliable agents.
