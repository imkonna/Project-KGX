# Project-KGX

GNNs for drug safety prediction. See [`docs/technical_report.tex`](docs/technical_report.tex).

## Setup

```bash
pip install -r requirements.txt
pip install PyTDC --no-deps
cp .env.example .env   # then paste your HF_TOKEN
```

## Run

**1. Sanity check (~20 min).** Open `notebooks/unified_benchmark.ipynb`. Near the top of section 6 there's a cell with two flags:

```python
SMOKE_TEST = True     # ← set this to True for the first run
USE_CHEMBERTA = True
```

Click **Run All**. When it finishes, look at `results/v3/` to confirm files were written.

**2. Full sweep (~8-12h).** Same notebook. Set `SMOKE_TEST = False` and click **Run All**. Safe to interrupt — completed runs are skipped on restart. Set `USE_CHEMBERTA = False` to save ~30-40% of the time.

**3. Conformer experiment (~30-60 min).** Open `notebooks/conformer_ensemble.ipynb` and click **Run All**. Needs step 2 to have finished ClinTox first.

**4. Commit.**

```bash
git add results/ models/
git commit -m "v3 results"
git push
```

## Team

Κωνσταντίνα Πάσχου · Γεώργιος Κεσογλίδης · Θεοχάρης Αγγελίδης
