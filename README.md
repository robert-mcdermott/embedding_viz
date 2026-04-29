## Lost Time Embedding Atlas

An interactive, WebGPU-first 3D semantic map of the vocabulary in `In-Search-of-Lost-Time-words.txt`.

### Build the embedding data

```bash
npm run build:data
```

The data builder uses `BAAI/bge-small-en-v1.5`, projects the normalized embeddings into 3D with UMAP, clusters the full embedding vectors with MiniBatchKMeans, and writes `public/data/embedding-map.json`.

### Run the browser app

```bash
npm run dev
```

Then open the Vite URL printed in the terminal, usually `http://127.0.0.1:5173/`.

### Useful options

```bash
uv run scripts/build_embeddings.py --device cpu
uv run scripts/build_embeddings.py --max-words 5000
uv run scripts/build_embeddings.py --clusters 30 --force
```
