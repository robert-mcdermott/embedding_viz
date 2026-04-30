## Embedding Atlas

An interactive, WebGPU-first 3D semantic map for word-level or sentence-level text embeddings. The included manifest ships with two datasets from *In Search of Lost Time*: full sentences extracted from the PDF and the original word list.

### Build The Included Data

```bash
npm run build:sentences
npm run build:words
```

Or rebuild both:

```bash
npm run build:all-data
```

The builder uses `BAAI/bge-small-en-v1.5`, projects normalized embeddings into 3D with UMAP, clusters the full embedding vectors with MiniBatchKMeans, and writes browser-ready JSON files under `public/data/`.

### Run The Browser App

```bash
npm run dev
```

Then open the Vite URL printed in the terminal, usually `http://127.0.0.1:5173/`.

### Build Your Own Dataset

Sentence atlas from a PDF or plain text file:

```bash
uv run scripts/build_embeddings.py --unit sentences --input my-book.pdf --output public/data/my-book-sentences.json --title "My Book Sentences"
```

Word atlas from a newline-delimited word list:

```bash
uv run scripts/build_embeddings.py --unit words --input words.txt --output public/data/my-words.json --title "My Words"
```

Add new datasets to `public/data/manifest.json` so they appear in the app's dataset selector:

```json
{
  "id": "my-book-sentences",
  "label": "My Book Sentences",
  "unit": "sentences",
  "url": "/data/my-book-sentences.json"
}
```

Useful builder options:

```bash
uv run scripts/build_embeddings.py --unit sentences --input my-book.pdf --dry-run
uv run scripts/build_embeddings.py --unit sentences --input my-book.pdf --start-at "Chapter 1" --end-before "Appendix"
uv run scripts/build_embeddings.py --unit words --input words.txt --max-items 5000
uv run scripts/build_embeddings.py --unit sentences --input my-book.pdf --clusters 40 --device cpu
```
