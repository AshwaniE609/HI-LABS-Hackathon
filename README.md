# Provider Specialty Standardization

Standardizes free-text healthcare provider specialties to official NUCC (National Uniform Claim Committee) taxonomy codes using a multi-method matching pipeline.

## Solution Pipeline

### 1. **Preprocessing**
- Standardizes input text (lowercase, removes extra whitespace)
- Expands medical abbreviations (cardio → cardiology, ENT → otolaryngology)
- Removes stopwords and special characters
- Detects compound specialties (e.g., "Obstetrics and Gynecology")
- Fixes common misspellings

### 2. **Multi-Method Matching** (Sequential Fallback)
- **Exact Match**: Direct match with NUCC display names (98% confidence)
- **Fuzzy Match**: Token-set ratio matching for minor variations (90% confidence)
- **Semantic Match**: Sentence transformers for similar meanings (85% confidence)
- **Multi-Specialty Match**: Splits and matches compound specialties (95% confidence)
- **Fallback Match**: Last-resort fuzzy matching for partial matches (50% confidence)

### 3. **Confidence Calibration**
- Applies isotonic regression to align confidence scores with actual accuracy
- Improves reliability of low-confidence predictions
- Adjusts scores based on the matching method

### 4. **Junk Classification**
- Identifies unmappable specialties (empty, <3 characters, no match found)
- Flags entries with low confidence for manual review

### 5. **Post-Processing**
- Collects alternative NUCC codes with confidence ≥ 0.6
- Consolidates results into a structured format
- Generates a summary with the primary code and alternatives

## Results

| Metric | Value |
|--------|-------|
| Total Processed | 10,050 |
| Successfully Mapped | 9,547 (95%) |
| Junk Records | 503 (5%) |
| Avg Calibrated Confidence | 0.909 |
| Method Distribution | Fuzzy: 4,874 \| Exact: 3,586 \| Semantic: 1,043 |

## Tech Stack

- **Python 3.x** - Core processing
- **RapidFuzz** - String similarity
- **SentenceTransformers** - Semantic embeddings
- **PyTorch** - Embedding operations
- **Scikit-learn** - Isotonic regression
- **Pandas** - Data manipulation

## Usage

from standardizer import ProviderSpecialtyStandardizer

Load data
nucc_df = pd.read_csv('nucc_taxonomy_master.csv')
input_df = pd.read_csv('input_specialties.csv')

Initialize and run
standardizer = ProviderSpecialtyStandardizer(nucc_df)
output_df = standardizer.standardize(input_df, specialty_column='rawspecialty')

Save results
output_df.to_csv('standardized_output.csv', index=False)

text

## Outputs

- `standardized_CORRECTED.csv` - Full results with confidence and alternatives
- `standardized_SUMMARY.csv` - Consolidated view (specialty, codes, confidence, junk flag)
