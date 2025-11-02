# Medical Specialty Standardization Pipeline

## Overview

This repository contains a **Medical Specialty Standardization System** that maps raw medical specialty names to standardized NUCC (National Uniform Claim Committee) taxonomy codes. The system uses a multi-layered matching approach combining exact matching, fuzzy matching, semantic analysis, and fallback mechanisms to ensure robust specialty classification.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Repository Setup](#repository-setup)
3. [Required Datasets](#required-datasets)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Processing Stages](#processing-stages)
6. [Configuration & Thresholds](#configuration--thresholds)
7. [Output Format](#output-format)
8. [Usage Examples](#usage-examples)
9. [Performance Metrics](#performance-metrics)

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- Required Libraries (see Installation below)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd specialty-standardization
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install required packages:
   ```bash
   pip install rapidfuzz sentence-transformers torch scikit-learn pandas numpy
   ```

3. **Download the Notebook**
   - Download `Final_submission.ipynb` from the repository
   - Open it in Jupyter Notebook or Google Colab

---

## Repository Setup

### Directory Structure

```
specialty-standardization/
├── Final_submission.ipynb          # Main execution notebook
├── input_specialties.csv           # Input specialty names (required)
├── NUC_tech_economy.csv            # NUCC taxonomy reference (required)
├── output_standardized_CORRECTED.csv # Full output with all columns
├── output_standardized_SUMMARY.csv  # Simplified summary output
└── README.md                        # This file
```

---

## Required Datasets

### 1. Input Specialties Dataset: `input_specialties.csv`

This CSV file contains the raw medical specialty names that need to be standardized.

**Column Name:** `raw_specialty`

**Example Format:**
```csv
raw_specialty
ACUPUNCTURE
ADOLESCENT MEDICINE
ALLERGY & IMMUNOLOGY
ANATOMIC & CLINICAL PATHOLOGY
ANESTHESIOLOGY
APPLIED BEHAVIORAL ANALYSIS (ABA)
AUDIOLOGY
BARIATRIC SURGERY
```

**Important:** Make sure the column is named exactly `raw_specialty` for the pipeline to work correctly.

### 2. NUCC Taxonomy Reference: `NUC_tech_economy.csv`

This dataset contains the standardized NUCC taxonomy codes and descriptions. It should include:

**Required Columns:**
- `Code` - NUCC taxonomy code (e.g., "207Q00000X")
- `Classification` - Specialty classification name
- `Specialization` - Detailed specialization

**Example Format:**
```csv
Code,Classification,Specialization
207Q00000X,Dermatology,Dermatology
207R00000X,Nephrology,Nephrology
207V00000X,Rheumatology,Rheumatology
```

**How to Locate:**
- The file should be in the same directory as the notebook
- Total expected records: approximately 879 NUCC entries
- This is the authoritative reference for medical specialties

---

## Pipeline Architecture

The Medical Specialty Standardization Pipeline operates through a **cascading multi-stage matching system**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT SPECIALTIES                             │
│                  (raw_specialty column)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: DATA PREPROCESSING                         │
│  • Whitespace normalization                                      │
│  • Case conversion (lowercase)                                   │
│  • Special character handling                                    │
│  • Medical vocabulary extraction                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: MEDICAL VOCABULARY BUILD                   │
│  • Extract medical terms from NUCC data                          │
│  • Create keyword dictionary                                     │
│  • Establish term synonyms                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          STAGE 3: MATCHING STRATEGY (Cascading)                 │
│                                                                  │
│  3.1 EXACT MATCHING                                             │
│      └─ Direct string match (confidence: 0.95)                  │
│                                                                  │
│  3.2 KEYWORD MATCHING & SEMANTIC ANALYSIS                       │
│      └─ Term matching in NUCC descriptions                      │
│      └─ Medical term extraction & matching                      │
│      └─ Semantic embeddings (Sentence-BERT)                     │
│      └─ Confidence scores computed via calibration              │
│                                                                  │
│  3.3 FUZZY MATCHING (Levenshtein Distance)                      │
│      └─ String similarity ratio calculation                      │
│      └─ Threshold: 0.7 (confidence: 0.9)                        │
│      └─ Handles typos & variations                              │
│                                                                  │
│  3.4 FALLBACK MATCHING                                          │
│      └─ Applied if other methods fail                           │
│      └─ Uses highest similarity available                       │
│      └─ Lower confidence threshold (0.4275 base)                │
│                                                                  │
│  3.5 NO MATCH / JUNK CLASSIFICATION                             │
│      └─ Assigned when no suitable match found                   │
│      └─ Flagged for manual review                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 4: CONFIDENCE CALIBRATION                       │
│  • Adjust raw confidence scores                                  │
│  • Apply method-specific calibration weights:                   │
│      - Exact Match: 0.95                                        │
│      - Fuzzy Match: 0.90                                        │
│      - Semantic Match: 0.83                                     │
│      - Fallback Match: 0.43                                     │
│  • Generate final calibrated confidence scores                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          STAGE 5: ALTERNATIVE CODES GENERATION                  │
│  • Extract top 5 alternative NUCC codes                         │
│  • Calculate scores for each alternative                        │
│  • Threshold filter: ≥ 0.6 for inclusion in results            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 6: VALIDATION & METRICS COMPUTATION             │
│  • Calculate validation metrics                                  │
│  • Assess mapping success rates                                  │
│  • Identify low-confidence records                              │
│  • Generate junk records report                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                             │
│  • Full detailed output CSV                                     │
│  • Summary output CSV (4 key columns)                           │
│  • Validation metrics report                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Processing Stages

### Stage 1: Data Preprocessing

**Purpose:** Clean and normalize input specialty names for consistent matching.

**Operations:**
```python
# Step 1.1: Whitespace Normalization
raw_input = "  DERMATOLOGY  AND   SURGERY  "
normalized = raw_input.strip().replace(multiple_spaces, single_space)
# Output: "DERMATOLOGY AND SURGERY"

# Step 1.2: Case Conversion
converted = normalized.lower()
# Output: "dermatology and surgery"

# Step 1.3: Special Character Handling
cleaned = converted.replace("&", "and").replace("/", " or ")
# Output: "dermatology and surgery"

# Step 1.4: Duplicate Term Removal
final = remove_duplicate_words(cleaned)
# Output: "dermatology and surgery"
```

**Preprocessing Class Methods:**
- `preprocess_text(text)` - Main preprocessing function
- `normalize_whitespace(text)` - Handle spacing
- `handle_special_chars(text)` - Manage symbols
- `expand_abbreviations(text)` - Convert acronyms

---

### Stage 2: Medical Vocabulary Construction

**Purpose:** Build a reference vocabulary from NUCC data for intelligent matching.

**Process:**

1. **Extract Medical Terms from NUCC Data**
   ```
   NUCC Code: 207Q00000X
   Classification: "Dermatology"
   Specialization: "Dermatology"

   Extracted Terms:
   - "dermatology"
   - "derm"
   - "skin"
   ```

2. **Build Synonym Dictionary**
   ```python
   medical_vocabulary = {
       "dermatology": ["derm", "skin specialist", "dermatologist"],
       "oncology": ["cancer", "tumor", "oncologist"],
       "cardiology": ["heart", "cardiac", "cardiologist"],
   }
   ```

3. **Create Keyword Maps**
   - Term → NUCC Code mapping
   - Alias → Primary Term mapping
   - Specialization → Classification mapping

**Vocabulary Building Methods:**
- `build_medical_vocabulary()` - Main construction
- `extract_terms(text)` - Term extraction
- `create_synonym_map()` - Alias creation
- `validate_vocabulary()` - Quality checks

---

### Stage 3: Multi-Layered Matching Strategy

#### 3.1 Exact Matching
- **Confidence:** 0.95
- **Method:** Direct string comparison after preprocessing
- **Best For:** Clean, standardized inputs

```python
if preprocessed_input == preprocessing(nucc_description):
    method = "exact_match"
    confidence = 0.95
```

#### 3.2 Keyword Matching & Semantic Analysis

**Keyword Matching:**
- Searches for preprocessed input in medical vocabulary
- Checks medical terms against NUCC descriptions
- Pattern matching on common medical terminology

**Semantic Analysis:**
- Uses Sentence-BERT transformer model (`all-MiniLM-L6-v2`)
- Converts input and NUCC descriptions to embeddings
- Computes cosine similarity between vectors
- Confidence ranges: 0.4 - 0.95 (calibrated)

```python
# Semantic matching example
input_embedding = model.encode("dermatology and skin care")
nucc_embedding = model.encode("Dermatology specialty")
similarity = cosine_similarity(input_embedding, nucc_embedding)
# similarity ≈ 0.83 → confidence = 0.83 (calibrated)
```

#### 3.3 Fuzzy Matching (Levenshtein Distance)
- **Threshold:** 0.7 (string similarity ratio)
- **Confidence:** 0.90 (if threshold met)
- **Best For:** Typos, spelling variations, abbreviations

```python
from rapidfuzz import fuzz
ratio = fuzz.token_sort_ratio(input, nucc_description) / 100

if ratio >= 0.7:
    method = "fuzzy_match"
    confidence = 0.90
```

**Example:**
```
Input: "DERMATOLOGY"
NUCC: "Dermatology"
Ratio: 1.0 (100% match) → Assigned to fuzzy_match
Confidence: 0.90
```

#### 3.4 Fallback Matching
- Applied when other methods fail to reach confidence threshold
- Selects highest similarity available
- Lower confidence score reflects uncertainty

```python
if no_match_found:
    best_match = select_highest_similarity()
    method = "fallback_match"
    confidence = 0.43  # Base fallback confidence
```

#### 3.5 No Match / Junk Classification
- Assigned when similarity falls below all thresholds
- Marked for manual review
- Primary Code: "JUNK"
- Confidence: 0.0

---

### Stage 4: Confidence Calibration

**Purpose:** Normalize confidence scores based on matching method.

**Calibration Weights by Method:**

| Method | Base Confidence | Calibration Weight | Final Range |
|--------|-----------------|-------------------|------------|
| Exact Match | 0.95 | 1.0 | 0.95 |
| Fuzzy Match | 0.90 | 1.0 | 0.70 - 0.90 |
| Semantic Match | Raw score | 0.8307 | 0.33 - 0.83 |
| Fallback Match | Raw score | 0.4275 | 0.04 - 0.43 |

**Calibration Formula:**
```
Calibrated_Confidence = Raw_Confidence × Method_Weight
```

**Example:**
```
Semantic similarity: 0.95
Method weight: 0.8307
Calibrated confidence: 0.95 × 0.8307 = 0.789
```

---

### Stage 5: Alternative Codes Generation

**Purpose:** Provide multiple valid NUCC codes when primary mapping has alternatives.

**Process:**

1. **Collect Top Alternatives**
   - Get top 5 NUCC codes by similarity score
   - Exclude primary code (already assigned)

2. **Score Filtering**
   - Include alternatives only if score ≥ 0.6
   - Lower threshold variants for edge cases

3. **Output Format**
   ```
   Primary_Code: 207Q00000X
   Primary_Confidence: 0.95

   Alternative_Code_1: 207N00000X (score: 0.87)
   Alternative_Code_2: 207U00000X (score: 0.78)
   Alternative_Code_3: 207S00000X (score: 0.65)
   (Alternative_Code_4 & 5 not included - score < 0.6)
   ```

---

### Stage 6: Validation & Metrics Computation

**Validation Metrics Calculated:**

| Metric | Description | Calculation |
|--------|-------------|-------------|
| total_records | Total input records processed | len(input_df) |
| junk_records | Records with no valid match | count(Primary_Code == "JUNK") |
| mapped_records | Successfully mapped records | total - junk |
| mapping_success_rate | % of records successfully mapped | (mapped / total) × 100 |
| junk_percentage | % of failed mappings | (junk / total) × 100 |
| low_confidence_count | Records below confidence threshold (0.6) | count(confidence < 0.6) |
| low_confidence_percentage | % of low confidence records | (low_conf / mapped) × 100 |
| avg_calibrated_confidence | Mean calibrated confidence score | mean(calibrated_confidence) |
| avg_original_confidence | Mean raw confidence score | mean(original_confidence) |
| confidence_improvement | Confidence change post-calibration | avg_calibrated - avg_original |
| method_distribution | Count by matching method | dict of method → count |
| multi_specialty_count | Records with multiple specialties | count('|' in Specialty) |
| multi_specialty_avg_confidence | Avg confidence for multi-specialty records | filtered mean |

**Example Metrics Output:**
```
total_records: 10050
mapped_records: 9547
junk_records: 503
mapping_success_rate: 95.0%
junk_percentage: 5.0%
avg_calibrated_confidence: 0.909
method_distribution: {
    'exact_match': 3586,
    'fuzzy_match': 4874,
    'semantic_match': 1043,
    'fallback_match': 44,
    'no_match': 401,
    'empty_input': 102
}
```

---

## Configuration & Thresholds

### Key Thresholds

```python
# Stage 2: Preprocessing
MIN_WORD_LENGTH = 2              # Minimum characters for a word
MAX_TERM_LENGTH = 50             # Maximum term length

# Stage 3: Matching Thresholds
EXACT_MATCH_CONFIDENCE = 0.95    # Exact match confidence
FUZZY_THRESHOLD = 0.7            # Fuzzy match ratio threshold
FUZZY_CONFIDENCE = 0.90          # Confidence if fuzzy threshold met
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # Semantic match threshold
MIN_CONFIDENCE_FOR_ALTERNATIVE = 0.6 # Include alternative codes if >= 0.6

# Stage 4: Calibration Weights
CALIBRATION_WEIGHTS = {
    'exact_match': 1.0,           # No adjustment
    'fuzzy_match': 1.0,           # No adjustment
    'semantic_match': 0.8307,     # Reduce by 16.93%
    'fallback_match': 0.4275      # Reduce by 57.25%
}

# Stage 6: Validation
LOW_CONFIDENCE_THRESHOLD = 0.6   # Consider as low confidence if below
```

### Customizing Thresholds

To modify thresholds for your specific use case, edit the configuration section in the notebook:

```python
class Config:
    FUZZY_THRESHOLD = 0.75  # Increase for stricter matching
    MIN_CONFIDENCE_FOR_ALTERNATIVE = 0.7  # Require higher scores
    CALIBRATION_WEIGHTS = {...}  # Adjust per your needs
```

---

## Output Format

### Full Output: `output_standardized_CORRECTED.csv`

Comprehensive output with all matching details.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| raw_specialty | string | Original input specialty name |
| Preprocessed | string | Cleaned specialty name |
| Primary_Code | string | Assigned NUCC code (or "JUNK") |
| Primary_Description | string | NUCC description for primary code |
| Calibrated_Confidence | float | Adjusted confidence score (0.0-1.0) |
| Original_Confidence | float | Raw confidence before calibration |
| Method | string | Matching method used |
| Alternative_Code_1-5 | string | Top 5 alternative NUCC codes |
| Alternative_Score_1-5 | float | Confidence scores for alternatives |
| Is_Multi_Specialty | boolean | True if input contains multiple specialties |
| Processing_Time | float | Seconds to process this record |

**Example Row:**
```csv
raw_specialty,Preprocessed,Primary_Code,Primary_Description,Calibrated_Confidence,Method
"DERMATOLOGY AND ALLERGY","dermatology and allergy","207Q00000X","Dermatology",0.95,"exact_match"
```

### Summary Output: `output_standardized_SUMMARY.csv`

Simplified 4-column output for quick reference.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| raw_specialty | string | Original input specialty |
| NUCC_Codes | string | Assigned NUCC code(s), separated by \|  (pipe) |
| Confidence | float | Primary confidence score |
| Junk | boolean | True if marked as JUNK |

**Example Rows:**
```csv
raw_specialty,NUCC_Codes,Confidence,Junk
"DERMATOLOGY","207Q00000X",0.95,False
"UNKNOWN SPECIALTY","JUNK",0.0,True
"CARDIOLOGY AND SURGERY","207R00000X \| 208000000X",0.88,False
```

---

## Usage Examples

### Example 1: Basic Usage in Jupyter

```python
import pandas as pd
from Final_submission import ProviderSpecialtyStandardizer

# 1. Load NUCC reference data
nucc_df = pd.read_csv('NUC_tech_economy.csv')

# 2. Load input specialties
input_df = pd.read_csv('input_specialties.csv')

# 3. Initialize standardizer
standardizer = ProviderSpecialtyStandardizer(nucc_df)

# 4. Run standardization
output_df = standardizer.standardize(
    input_df,
    specialty_column='raw_specialty'
)

# 5. Generate metrics
metrics = standardizer.compute_validation_metrics(output_df)

# 6. Save outputs
output_df.to_csv('output_standardized_CORRECTED.csv', index=False)

print("✓ Processing complete!")
print(f"Success rate: {metrics['mapping_success_rate']}%")
```

### Example 2: Handling Multi-Specialty Records

```python
# Input with multiple specialties
input_data = pd.DataFrame({
    'raw_specialty': [
        'DERMATOLOGY & ALLERGY',
        'CARDIOLOGY/CARDIAC SURGERY',
        'PEDIATRICS AND PSYCHIATRY'
    ]
})

output = standardizer.standardize(input_data, 'raw_specialty')

# Output shows multiple codes separated by |
print(output[['raw_specialty', 'NUCC_Codes']])
```

### Example 3: Custom Threshold Configuration

```python
# Modify matching thresholds
standardizer.config.FUZZY_THRESHOLD = 0.8  # Stricter matching
standardizer.config.MIN_CONFIDENCE_FOR_ALTERNATIVE = 0.7

# Re-run with new thresholds
output_df = standardizer.standardize(input_df, 'raw_specialty')
```

### Example 4: Filtering Results

```python
# Get only high-confidence mappings
high_confidence = output_df[output_df['Calibrated_Confidence'] >= 0.85]

# Get unmapped records (JUNK)
unmapped = output_df[output_df['Primary_Code'] == 'JUNK']

# Get records with alternative matches
with_alternatives = output_df[output_df['Alternative_Code_1'].notna()]
```

---

## Performance Metrics

### Expected Performance on Standard Datasets

Based on testing with ~10,000 medical specialty records:

| Metric | Value |
|--------|-------|
| Total Records Processed | 10,050 |
| Successfully Mapped | 9,547 (95.0%) |
| Junk Records | 503 (5.0%) |
| Average Calibrated Confidence | 0.909 |
| Exact Matches | 3,586 (35.7%) |
| Fuzzy Matches | 4,874 (48.4%) |
| Semantic Matches | 1,043 (10.4%) |
| Fallback Matches | 44 (0.4%) |
| Multi-Specialty Records | 3,673 (36.5%) |
| Low Confidence Records (< 0.6) | 44 (0.46%) |
| Processing Time | ~2-5 seconds per 1,000 records |

### Performance Optimization Tips

1. **Reduce Dataset Size:** Process in batches of 5,000-10,000 records
2. **Cache Embeddings:** Store semantic embeddings for repeated lookups
3. **Use GPU:** Enable CUDA for faster embedding generation
4. **Adjust Thresholds:** Stricter thresholds reduce processing overhead

---

## Troubleshooting

### Issue: Column Name Error
```
Error: KeyError: 'raw_specialty'
```
**Solution:** Ensure your input CSV has a column named exactly `raw_specialty`

### Issue: Missing NUCC Reference Data
```
Error: FileNotFoundError: NUC_tech_economy.csv
```
**Solution:** Place `NUC_tech_economy.csv` in the same directory as the notebook

### Issue: Low Matching Success Rate
```
mapping_success_rate: 60%
```
**Solutions:**
- Verify input data format and spellings
- Adjust thresholds to be more lenient
- Check NUCC reference data completeness
- Review junk records for patterns

### Issue: GPU Memory Exhaustion
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Use CPU instead: `device='cpu'`
- Reduce batch size
- Clear GPU cache between runs

---

## References & Documentation

- **NUCC Taxonomy:** [National Uniform Claim Committee](https://www.nucc.org/)
- **Sentence-BERT:** [Sentence Transformers Documentation](https://www.sbert.net/)
- **RapidFuzz:** [Fuzzy String Matching Library](https://github.com/maxbachmann/RapidFuzz)
- **scikit-learn:** [Machine Learning Library](https://scikit-learn.org/)

---

## License & Usage

This project is provided as-is for medical specialty standardization purposes. For questions or support, please refer to the attached documentation or contact the development team.

**Last Updated:** November 2025
**Version:** 1.0.0
