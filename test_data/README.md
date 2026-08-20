# Canonical Test Asset & Data Policy

This directory contains test inputs and runtime outputs for the Chameleon project.

---

## Directory Structure

```text
test_data/
├── inputs/
│   ├── performer/   # Current performer motion-validation inputs
│   └── identity/    # Current reference identity/body assets (front, back, profiles)
├── outputs/         # Runtime-generated test & validation artifacts (git-ignored)
└── README.md
```

---

## Data Policy & Guidelines

### 1. Input Assets (`test_data/inputs/`)

- `inputs/performer/`: Contains only active performer motion capture validation images and videos.
- `inputs/identity/`: Contains only active target reference identity/body assets.
- Input directories must only hold active canonical datasets provided explicitly for testing.

### 2. Generated Outputs (`test_data/outputs/`)

- All runtime-generated artifacts (e.g., debug overlays, validation reports, side-by-side renders, JSON reports) must be written under `test_data/outputs/` or subdirectories inside `outputs/`.
- All contents of `test_data/outputs/` are ignored by Git (except `.gitkeep`). Generated files must never be committed to source control.

### 3. Canonical Asset Rule

- Scripts must receive **explicit asset paths** via command-line arguments (e.g., `--image test_data/inputs/performer/...`, `--output-dir test_data/outputs/...`).
- Scripts must **never**:
  - Silently fall back to hardcoded image filenames.
  - Automatically select the first file in a directory.
  - Reuse generated outputs or historical artifacts as input.
  - Fail silently if an input image is missing; scripts must fail clearly with an explicit error message.

### 4. Replacement Rule

- When new canonical test data is provided, obsolete input assets must be explicitly removed or archived outside the active repository dataset.
- There must be **exactly one clearly identifiable active dataset** in `inputs/`.
