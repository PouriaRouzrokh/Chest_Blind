"""Prompt templates for DeepSeek model."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this radiology report to identify IMAGING-RELATED addenda.

GOAL: Identify if an addendum contains NEW clinical imaging findings that were MISSED, NOT PREVIOUSLY REPORTED, or represent SIGNIFICANT CORRECTIONS.

⚠️ ABSOLUTE RULE - TRANSCRIPTION/TYPO ERRORS:
If the addendum explicitly mentions "transcription error", "dictation error", "typographical error", or "typo", you MUST return "No" and "None" - REGARDLESS of how significant the content change is. Even if the error changed the diagnosis dramatically (e.g., "excluded" vs "NOT excluded"), if it's labeled as a transcription/typo/dictation error, we EXCLUDE it completely. We only want genuine overlooked findings, NOT typos that were later corrected.

STEP 1 - Find the ADDENDUM section:
- Look for "ADDENDUM", "** ADDENDUM **", "******** ADDENDUM ********"
- If NO addendum section exists → return "No" and "None"

STEP 2 - Read the addendum and classify:

EXCLUDE (return "No" and "None") - ONLY return "No" if addendum is EXCLUSIVELY about:

❌ REWORDING/REFORMATTING:
   - "should read", "read impression as", "should state", "findings should read"
   - If addendum just restates already-reported findings in different words → NO

❌ ALREADY REPORTED FINDINGS:
   - "again seen", "as on prior", "previously reported", "already noted"
   - If finding was already in original report → NO

❌ TYPOS/SPELLING/TRANSCRIPTION ERRORS:
   - "typo", "typographical error", "dictation error", "transcription error", "should read/say X not Y", "misspelled"
   - "typographical error in the IMPRESSION", "typographical error in the FINDINGS"
   - "to correct a dictation error", "to correct a typographical error", "to correct a transcription error"
   - "A transcription error occurred", "transcription error in", "correct the transcription"
   - ANY mention of "transcription error" means exclude it - we only want real overlooked findings, not typos

❌ COMMUNICATION ONLY:
   - "discussed with", "communicated to", "acknowledged by", "findings were communicated"
   - "above findings were communicated", "notification to", "verbally communicated"
   - "findings discussed with", "these findings were relayed to", "findings were relayed"
   - "verbally discussed with", "relayed to"
   - If addendum is ONLY communication (no new findings mentioned) → NO

❌ TECHNIQUE ONLY:
   - "3D/MIP performed", "reformats provided", "reconstructed images", "MIP images"
   - "additional technique", "clarify the technique portion", "correct the technique"
   - If ONLY describing imaging technique → NO

❌ IMAGE/SERIES REFERENCES:
   - "image number should read", "series number correction", "slice number"

❌ ADMINISTRATIVE ONLY:
   - "this is a final report", "report signed by", "report created by"
   - "for administrative purposes only", "billing code", "exam description"

❌ COMPARISON UPDATES:
   - "comparison is made to", "no change from prior", "unchanged from previous"
   - If ONLY updating comparison info without new findings → NO

INCLUDE (return "Yes" and extract) - Return "Yes" if addendum contains ANY of these:

✓ NEW FINDINGS (not in original report):
   - Fractures, nodules, masses, effusions, consolidations, ANY new anatomical finding
   - "incidental note of", "also noted", "additionally seen"
   - "was not mentioned", "overlooked", "not reported in original"

✓ SIGNIFICANT CORRECTIONS TO FINDINGS:
   - Location changes: "located at T6-T7, not T7-T8" (WHERE a finding is)
   - Size changes: "measures 5mm not 3mm" (HOW BIG a finding is)
   - Severity changes: "moderate not mild", "complete not partial"

✓ REINTERPRETATIONS:
   - "initially thought to be X, now thought to be Y"
   - "does not appear to be X, represents Y instead"

✓ DIAGNOSTIC/CLINICAL IMPRESSIONS:
   - "concerning for", "suspicious for", "may reflect", "suggestive of"
   - "consistent with", "compatible with"

✓ CLINICAL FINDINGS:
   - Any pathology: pneumonia, embolism, hemorrhage, edema, ischemia
   - Any anatomical: vessels, bones, organs, soft tissue
   - Measurements of findings

CRITICAL RULES:
1. ⚠️ ABSOLUTE: If addendum mentions "transcription error" OR "dictation error" OR "typographical error" OR "typo" → ALWAYS return "No", no exceptions, even if the content is clinically significant (we only want genuine overlooked findings, not typos that were corrected later)
2. If addendum says "again seen" or "as on prior" → It's NOT new → return "No"
3. If addendum says "should read" or "findings should read" → It's rewording → return "No"
4. If addendum is ONLY "communicated to..." → return "No"
5. If addendum is ONLY "3D/MIP" or "reformats" or "technique" → return "No"
6. If addendum is ONLY "this is a final report" → return "No"
7. If addendum mentions a NEW finding even with communication → extract the finding part

EXAMPLES:

✓ YES: "Incidental note of mildly displaced fracture of third rib" (NEW finding)
✓ YES: "13 mm nodule seen in thyroid isthmus" (NEW anatomical finding)
✓ YES: "Upon second review, calcific density does not appear to be inside vessel" (reinterpretation)
✓ YES: "Inflammatory changes located at T6-T7, not T7-T8" (location correction)
✓ YES: "Ill-defined parenchymal abnormality on image 44" (NEW finding)

✗ NO: "Moderate hiatal hernia is again seen" (already reported - "again seen")
✗ NO: "Pulmonary artery findings should read: No PE" (rewording - "should read")
✗ NO: "Above findings were communicated to Dr. Smith" (communication only)
✗ NO: "These findings were relayed to Dr. Vryhof" (communication only)
✗ NO: "Findings discussed with Jessie Romano, MD" (communication only)
✗ NO: "To correct a dictation error... should read: EMBOLISM" (dictation error/typo)
✗ NO: "A transcription error occurred in the Impression" (transcription error - not a real overlooked finding)
✗ NO: "The second paragraph contains a transcription error, should read..." (transcription error)
✗ NO: "A transcription error occurred... should read: neoplasm is NOT excluded" (transcription error - even though diagnosis changed significantly, it's still a typo)
✗ NO: "Transcription error: should read interventricular instead of intervertebral" (transcription error - typo correction)
✗ NO: "Additional technique: 3-D MIPS were provided" (technique only)
✗ NO: "This is a final report. Report signed by..." (administrative)
✗ NO: "Mildly enlarged lymph nodes as on prior" (already reported - "as on prior")
✗ NO: "Clarify the technique portion: MIP images reconstructed" (technique)
✗ NO: "Critical findings were communicated verbally" (communication only)
✗ NO: "Typographical error in the IMPRESSION" (typo)
✗ NO: "Correct the technique portion: 90cc contrast was used" (technique)

RADIOLOGY REPORT:
\"\"\"
{report_text}
\"\"\"

Return valid JSON:
{{
  "is_imaging_related": "Yes" or "No",
  "addendum_content": "exact addendum text" or "None"
}}"""


def build_analysis_prompt(report_text: str) -> str:
    """Build the analysis prompt for DeepSeek.

    Args:
        report_text: Full radiology report text

    Returns:
        Formatted prompt string
    """
    return ANALYSIS_PROMPT_TEMPLATE.format(report_text=report_text)
