## Embedding Atlas

Embedding Atlas is a WebGPU-first browser app for exploring semantic embeddings as an interactive 3D point cloud. It can visualize either word-level embeddings from a newline-delimited word list or sentence-level embeddings extracted from a PDF or plain text file.

The repo includes two example datasets from `In-Search-of-Lost-Time.pdf`:

- `Lost Time Sentences`: 29,805 full-sentence embeddings.
- `Lost Time Words`: 29,063 word embeddings from `In-Search-of-Lost-Time-words.txt`.

Each point is colored by semantic cluster. Hovering reveals the text represented by a point, clicking selects it, and the detail panel shows its cluster plus nearest semantic neighbors.

### Requirements

- Node.js and npm for the browser app.
- `uv` for Python dependency management and data generation.
- A modern browser with WebGPU support, such as current Chrome or Edge.
- Internet access on the first embedding build so Sentence Transformers can download the model.

PDF extraction uses Poppler's `pdftotext` when it is installed, and falls back to `pypdf` from the Python environment.

### Quick Start

Install dependencies after cloning:

```bash
npm install
uv sync
```

Run the browser app:

```bash
npm run dev
```

Open the local URL printed by Vite, usually `http://127.0.0.1:5173/`. The app loads datasets listed in `public/data/manifest.json`.

### Build The Included Examples

Build the default sentence dataset from the included PDF:

```bash
npm run build:sentences
```

Build the word dataset from the included word list:

```bash
npm run build:words
```

Build both:

```bash
npm run build:all-data
```

`npm run build:data` is an alias for the sentence dataset because sentence-level visualization is the default example.

### Generate Your Own Sentence Atlas

Use `--unit sentences` for a PDF or plain text file. Output files should usually go under `public/data/` so the browser app can fetch them.

```bash
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --output public/data/my-book-sentences.json \
  --title "My Book Sentences"
```

Before spending time on embedding, run a dry run to check the extracted sentences:

```bash
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --dry-run
```

For books with front matter, tables of contents, appendices, or license text, trim the source text with marker strings:

```bash
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --output public/data/my-book-sentences.json \
  --title "My Book Sentences" \
  --start-at "Chapter 1" \
  --end-before "Appendix"
```

Use `--start-at` to keep the marker text, `--start-after` to discard the marker text too, and `--end-before` to stop before a later marker.

### Generate Your Own Word Atlas

Use `--unit words` with a UTF-8 text file containing one word or phrase per line:

```txt
memory
cathedral
jealousy
time
```

Build the dataset:

```bash
uv run scripts/build_embeddings.py \
  --unit words \
  --input words.txt \
  --output public/data/my-words.json \
  --title "My Words"
```

Duplicate entries are removed case-insensitively.

### Add A Dataset To The App

After generating a JSON file, add it to `public/data/manifest.json`:

```json
{
  "datasets": [
    {
      "id": "my-book-sentences",
      "label": "My Book Sentences",
      "unit": "sentences",
      "url": "/data/my-book-sentences.json",
      "default": true
    },
    {
      "id": "my-words",
      "label": "My Words",
      "unit": "words",
      "url": "/data/my-words.json"
    }
  ]
}
```

Only one dataset should have `"default": true`. You can also open a dataset directly with a query string:

```text
http://127.0.0.1:5173/?dataset=my-words
```

### Builder Options

The builder uses `BAAI/bge-small-en-v1.5` by default, projects normalized embeddings into 3D with UMAP, clusters the original embedding vectors with MiniBatchKMeans, and writes browser-ready JSON.

Common options:

```bash
# Limit size while experimenting
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --output public/data/my-book-sentences.json \
  --max-items 5000

# Use a different Sentence Transformers model
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --output public/data/my-book-sentences.json \
  --model sentence-transformers/all-MiniLM-L6-v2

# Choose the number of semantic clusters
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --output public/data/my-book-sentences.json \
  --clusters 40

# Force CPU, Apple Silicon MPS, or CUDA when available
uv run scripts/build_embeddings.py \
  --unit sentences \
  --input my-book.pdf \
  --output public/data/my-book-sentences.json \
  --device cpu

# Recompute embeddings even if a cache exists
uv run scripts/build_embeddings.py \
  --unit words \
  --input words.txt \
  --output public/data/my-words.json \
  --force
```

Full option list:

```bash
uv run scripts/build_embeddings.py --help
```

### Output And Caching

Generated browser datasets live wherever you pass `--output`; the app expects manifest URLs that are fetchable from Vite, so `public/data/*.json` is the simplest place.

If you omit `--output`, the builder writes to `public/data/embedding-map.json`. That is useful for quick experiments, but explicit output paths are safer when you are maintaining multiple datasets.

Embedding vectors are cached under `.cache/embeddings/`. Hugging Face model files are cached under `.hf-home/`. Both are ignored by git. If you change the source text, model, or unit type, the builder creates a new embedding cache key.

Large datasets are normal. The included sentence dataset is about 11 MB and the word dataset is about 4 MB.

### Browser Controls

- Dataset selector: switch between generated datasets.
- Search: find a word or sentence and click a result to focus it.
- Hover: show the point label and nearby semantic links.
- Click: select a point and show nearest semantic neighbors.
- Links toggle: turn nearest-neighbor lines on or off.
- Point Size and Glow sliders: tune the rendering.
- Cluster list: isolate a semantic cluster by color.
- Reset, pause motion, frame selection, and clear focus are available in the icon toolbar.

### Production Build

Build the frontend:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

### Troubleshooting

If the app cannot find a dataset, check that the JSON file exists under `public/data/` and that `public/data/manifest.json` points to it with a URL like `/data/my-file.json`.

If sentence extraction looks wrong, use `--dry-run` first and adjust `--start-at`, `--start-after`, `--end-before`, or `--min-sentence-words`.

If embedding generation is slow, try `--max-items` while experimenting, reduce `--batch-size`, or force the available accelerator with `--device mps` or `--device cuda`.

If the browser does not show WebGPU, use a current Chrome or Edge build and check the Renderer metric in the app header.
