import os
import plotly.io as pio
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from typing import List
import openai

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY
pio.renderers.default = 'iframe'
pd.options.plotting.backend = "plotly"


embedding_model = "text-embedding-ada-002"


def embedding_texts(texts: List[str]) -> List[str]:
    resp = openai.Embedding.create(input=texts, model=embedding_model)
    if isinstance(resp, dict):
        return [rec['embedding'] for rec in resp['data']]
    return []


def read_txt_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as fp:
        data = fp.read()

    return data


def split_texts(text: str) -> List[str]:
    return list(filter(lambda x: len(x) > 0, text.split('\n')))


def merge_split_text(text: str, num_token: int) -> List[str]:
    text = text.replace('\n', ' ')
    texts = [text[x:x+num_token] for x in range(0, len(text), num_token)]
    return texts


def plot_scatter(df: pd.DataFrame):
    fig = px.scatter(df, x="x", y="y", color='types',
                     text="short_text", hover_name="index", size_max=60)

    fig.update_traces(textposition='top center', textfont_size=8)
    fig.update_layout(
        height=800,
        title_text='Your title'
    )

    fig.show()


texts = []
types = []


df = pd.DataFrame(data={
    'text': texts,
    'embedding': embedding_texts(texts),
    'types': types,
}).sample(frac=1)  # shuffle


matrix = np.array(df.embedding.to_list())

# Use t-SNE to project the high-dimensional vectors into two dimensions
tsne = TSNE(n_components=2, perplexity=15, random_state=42,
            init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

df['x'] = vis_dims[:, 0]
df['y'] = vis_dims[:, 1]
df['short_text'] = df.text.str[:50]
df['index'] = df.index
