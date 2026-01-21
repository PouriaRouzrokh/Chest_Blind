"""Quick test of updated pipeline."""

import sys
sys.path.insert(0, 'src')

from ollama_client import OllamaClient

# Sample radiology report text
sample_report = """******** ADDENDUM #1 ********

Addendum: Upon further review, a 5mm nodule is identified in the right upper lobe
that was not mentioned in the original report. This requires follow-up imaging in
3-6 months.

Impression: New finding - right upper lobe nodule, 5mm.

Reported And Signed By: John Doe, MD

******** ORIGINAL REPORT ********

Technique: Chest CT with contrast

History: 65-year-old with chronic cough

Findings: The lungs are clear. No masses or consolidations. Heart size is normal.

Impression: No acute findings."""

print("Testing updated Ollama client with format='json'...")
print("=" * 60)

client = OllamaClient()

if not client.check_availability():
    print("❌ Ollama not available")
    sys.exit(1)

print("✓ Ollama is available")
print("\nAnalyzing sample report...")

result = client.analyze_report(sample_report)

print("\nResult:")
print(f"  Is Imaging Related: {result['is_imaging_related']}")
print(f"  Addendum Content: {result['addendum_content'][:200]}...")

if result['is_imaging_related'] == 'Yes':
    print("\n✓ Test PASSED - Correctly identified imaging-related addendum")
else:
    print("\n✗ Test FAILED - Should have identified as imaging-related")
