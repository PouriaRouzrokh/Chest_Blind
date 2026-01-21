# Chest Blind - Radiology Report Addendum Analysis

Automated analysis of radiology reports to identify imaging-related addenda using local LLMs via Ollama.

## Overview

This project processes radiology reports from chest CT scans to automatically identify and extract imaging-related addenda. It distinguishes between genuine overlooked findings and administrative corrections (transcription errors, technique notes, communication logs).

## Features

- **Batch Processing with Resume Capability**: Process large datasets in manageable batches with automatic checkpoint/resume
- **Two Operating Modes**:
  - **Test Mode**: Random sampling with timestamped outputs for validation
  - **Production Mode**: Sequential processing with resume capability
- **Advanced LLM Integration**: Uses GPT-OSS 20B via Ollama with configurable reasoning effort
- **Comprehensive Filtering**: Excludes transcription errors, technique notes, communication logs, and administrative addenda
- **Reasoning Capture**: Saves model's thinking process for transparency

## Requirements

- Python 3.8+
- Ollama (https://ollama.ai)
- GPT-OSS 20B model (`ollama pull gpt-oss:20b`)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:PouriaRouzrokh/Chest_Blind.git
cd Chest_Blind
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install requests
```

4. Install Ollama and pull the model:
```bash
# Install Ollama from https://ollama.ai
ollama pull gpt-oss:20b
```

## Configuration

Edit `src/config.py` to configure:

- **RUN_MODE**: `"test"` or `"production"`
- **BATCH_SIZE**: Number of rows per batch in production mode
- **MODEL_NAME**: Ollama model to use (default: `gpt-oss:20b`)
- **REASONING_EFFORT**: `"low"`, `"medium"`, or `"high"`

## Usage

### Test Mode (Random Sampling)
```python
# In src/config.py
RUN_MODE = "test"
NUM_ROWS = 20
RANDOM_SAMPLE = True
```

```bash
python src/main.py
```

### Production Mode (Resumable Processing)
```python
# In src/config.py
RUN_MODE = "production"
BATCH_SIZE = 50  # or None for all remaining
```

```bash
# First run - processes first 50 reports
python src/main.py

# Second run - automatically resumes from row 51
python src/main.py

# Continue running until all reports processed
```

### Interruption Handling

If processing is interrupted (Ctrl+C or system crash):
- Progress is automatically saved
- Next run resumes from last completed report
- No reports are skipped or duplicated

## Output

### Test Mode
- Output: `data/output/processed_reports_test_YYYYMMDD_HHMMSS.csv`
- Creates new file each run with timestamp

### Production Mode
- Output: `data/output/processed_reports_final.csv`
- Checkpoint: `data/output/checkpoint.json`
- Appends to existing file on resume

### Output Columns
Original columns plus:
- **Imaging Related Addendum**: Yes/No/Error
- **Imaging Addendum Content**: Extracted text or None
- **Model Reasoning**: GPT-OSS's thinking process

## Inclusion Criteria

**INCLUDED** (Imaging-Related):
- New findings missed in original report
- Previously unreported lesions or anatomical abnormalities
- Significant corrections to finding location, size, or severity
- Reinterpretations of imaging findings
- Clinical/diagnostic impressions from imaging

**EXCLUDED** (Not Imaging-Related):
- Transcription/dictation/typographical errors
- Technique notes (MIP, reformats, contrast details)
- Communication logs ("discussed with...", "findings relayed...")
- Administrative notes ("this is a final report")
- Already reported findings ("again seen", "as on prior")
- Rewording without new clinical information

## Project Structure

```
Chest_Blind/
├── src/
│   ├── config.py           # Configuration settings
│   ├── prompts.py          # LLM prompts
│   ├── ollama_client.py    # Ollama API interface
│   ├── checkpoint.py       # Resume/checkpoint management
│   └── main.py             # Main processing script
├── data/
│   ├── output/             # Generated output files
│   └── *.csv               # Input data (gitignored)
├── tests/
│   └── test_pipeline.py    # Basic pipeline test
└── README.md
```

## Model Performance

- **Accuracy**: ~95% on test samples (20/20 validation)
- **Speed**: ~27 seconds per report (GPT-OSS 20B, medium reasoning)
- **Transcription Error Exclusion**: 100% (strict enforcement)

## License

[Add your license here]

## Citation

[Add citation information if applicable]

## Contact

[Add contact information]
