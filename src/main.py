"""Main script for processing radiology reports with resume capability."""

import csv
import logging
import os
import sys
import random
from datetime import datetime
from typing import List, Tuple

import config
from ollama_client import OllamaClient
from checkpoint import CheckpointManager


def setup_logging(run_mode: str):
    """Configure logging to file and console.

    Args:
        run_mode: "test" or "production"
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(
        os.path.dirname(config.OUTPUT_DIR),
        f"processing_{run_mode}_{timestamp}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Logging initialized")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Run mode: {run_mode}")


def read_all_csv_rows(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """Read all CSV rows.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (header, all_data_rows)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)

    logging.info(f"Read {len(all_rows)} total data rows from {file_path}")
    return header, all_rows


def get_rows_for_test_mode(all_rows: List[List[str]]) -> List[List[str]]:
    """Get rows for test mode (random sampling).

    Args:
        all_rows: All data rows

    Returns:
        Selected rows for testing
    """
    if config.RANDOM_SAMPLE and config.NUM_ROWS < len(all_rows):
        random.seed(config.RANDOM_SEED)
        selected = random.sample(all_rows, config.NUM_ROWS)
        logging.info(f"Randomly sampled {config.NUM_ROWS} rows from {len(all_rows)} total rows (seed={config.RANDOM_SEED})")
    else:
        selected = all_rows[:config.NUM_ROWS]
        logging.info(f"Selected first {config.NUM_ROWS} rows")

    return selected


def get_rows_for_production_mode(all_rows: List[List[str]],
                                  start_index: int,
                                  batch_size: int = None) -> List[List[str]]:
    """Get rows for production mode (sequential processing with resume).

    Args:
        all_rows: All data rows
        start_index: Index to start from (0-indexed)
        batch_size: Number of rows to process (None for all remaining)

    Returns:
        Selected rows for processing
    """
    if start_index >= len(all_rows):
        logging.info(f"All {len(all_rows)} rows already processed")
        return []

    if batch_size is None:
        end_index = len(all_rows)
    else:
        end_index = min(start_index + batch_size, len(all_rows))

    selected = all_rows[start_index:end_index]
    logging.info(f"Processing rows {start_index + 1} to {end_index} (of {len(all_rows)} total)")

    return selected


def write_output_csv(header: List[str],
                     data_rows: List[List[str]],
                     results: List[Tuple[str, str, str]],
                     output_path: str,
                     append_mode: bool = False):
    """Write or append processed results to CSV.

    Args:
        header: CSV header row
        data_rows: Original data rows being processed
        results: List of (is_imaging_related, addendum_content, reasoning) tuples
        output_path: Output file path
        append_mode: If True, append to existing file; if False, create new file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mode = 'a' if append_mode else 'w'

    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Write header only if creating new file
        if not append_mode:
            new_header = header.copy()
            new_header.insert(9, "Imaging Related Addendum")
            new_header.insert(10, "Imaging Addendum Content")
            new_header.insert(11, "Model Reasoning")
            writer.writerow(new_header)

        # Write data rows
        for i, row in enumerate(data_rows):
            new_row = row.copy()
            is_related, content, reasoning = results[i]
            new_row.insert(9, is_related)
            new_row.insert(10, content)
            new_row.insert(11, reasoning)
            writer.writerow(new_row)

    logging.info(f"Output written to: {output_path} (mode: {'append' if append_mode else 'create'})")


def process_reports(client: OllamaClient,
                   data_rows: List[List[str]],
                   start_index: int = 0,
                   total_count: int = None) -> List[Tuple[str, str, str]]:
    """Process reports with progress tracking.

    Args:
        client: OllamaClient instance
        data_rows: Data rows to process
        start_index: Starting index for progress display (0 for first batch)
        total_count: Total number of reports in dataset (for progress display)

    Returns:
        List of (is_imaging_related, addendum_content, reasoning) tuples
    """
    results = []
    count = len(data_rows)

    if total_count is None:
        total_count = count

    logging.info(f"Processing {count} reports...")
    print(f"\nProcessing {count} reports (rows {start_index + 1} to {start_index + count} of {total_count} total)...\n")

    for i, row in enumerate(data_rows, start=1):
        try:
            # Get report text from column 9 (index 8)
            if len(row) <= config.REPORT_COLUMN_INDEX:
                logging.warning(f"Row {start_index + i}: Missing report text column")
                results.append(("Error", "Missing report text column", ""))
                print(f"[{start_index + i}/{total_count}] Processing... ✗ Error: Missing column")
                continue

            report_text = row[config.REPORT_COLUMN_INDEX]

            # Analyze report
            result = client.analyze_report(report_text)
            is_related = result['is_imaging_related']
            content = result['addendum_content']
            reasoning = result.get('reasoning', '')

            results.append((is_related, content, reasoning))

            # Progress indicator
            status_symbol = "✓" if is_related in ["Yes", "No"] else "✗"
            print(f"[{start_index + i}/{total_count}] Processing... {status_symbol} {is_related}")

            if is_related == "Error":
                logging.error(f"Row {start_index + i}: {content}")

        except KeyboardInterrupt:
            logging.warning(f"Processing interrupted by user at row {start_index + i}")
            print(f"\n⚠️ Processing interrupted. Processed {i - 1} rows in this batch.")
            # Return results processed so far
            return results
        except Exception as e:
            logging.error(f"Row {start_index + i}: Unexpected error: {e}")
            results.append(("Error", str(e), ""))
            print(f"[{start_index + i}/{total_count}] Processing... ✗ Error: {str(e)[:50]}")

    return results


def run_test_mode(client: OllamaClient, header: List[str], all_rows: List[List[str]]):
    """Run in test mode with random sampling and timestamped output.

    Args:
        client: OllamaClient instance
        header: CSV header row
        all_rows: All data rows
    """
    logging.info("=" * 60)
    logging.info("TEST MODE: Random sampling with timestamped output")
    logging.info("=" * 60)

    # Get random sample
    selected_rows = get_rows_for_test_mode(all_rows)

    # Process reports
    start_time = datetime.now()
    results = process_reports(client, selected_rows, start_index=0, total_count=len(selected_rows))
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Write output with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.OUTPUT_DIR, f"processed_reports_test_{timestamp}.csv")
    write_output_csv(header, selected_rows, results, output_path, append_mode=False)

    # Print summary
    print_summary(results, duration, output_path)


def run_production_mode(client: OllamaClient, header: List[str], all_rows: List[List[str]]):
    """Run in production mode with resume capability.

    Args:
        client: OllamaClient instance
        header: CSV header row
        all_rows: All data rows
    """
    logging.info("=" * 60)
    logging.info("PRODUCTION MODE: Sequential processing with resume capability")
    logging.info("=" * 60)

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(config.OUTPUT_DIR)

    # Check for existing progress
    processed_count = checkpoint_mgr.get_processed_count()
    total_rows = len(all_rows)

    if processed_count > 0:
        logging.info(f"Found existing progress: {processed_count}/{total_rows} rows already processed")
        print(f"\n✓ Resuming from row {processed_count + 1}")
    else:
        logging.info(f"Starting fresh processing of {total_rows} rows")
        print(f"\n✓ Starting fresh processing")

    # Get rows to process in this batch
    batch_rows = get_rows_for_production_mode(all_rows, processed_count, config.BATCH_SIZE)

    if not batch_rows:
        print(f"\n✓ All {total_rows} rows have been processed!")
        logging.info("All rows already processed")
        return

    # Process reports
    start_time = datetime.now()
    results = process_reports(client, batch_rows, start_index=processed_count, total_count=total_rows)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Write output (append if file exists)
    output_path = checkpoint_mgr.output_path
    append_mode = checkpoint_mgr.output_exists()
    write_output_csv(header, batch_rows, results, output_path, append_mode=append_mode)

    # Save checkpoint
    new_processed_count = processed_count + len(results)
    checkpoint_mgr.save_checkpoint(new_processed_count - 1, total_rows)
    logging.info(f"Checkpoint saved: {new_processed_count}/{total_rows} rows processed")

    # Print summary
    print_summary_production(results, duration, output_path, new_processed_count, total_rows)


def print_summary(results: List[Tuple[str, str, str]], duration: float, output_path: str):
    """Print summary for test mode.

    Args:
        results: Processing results
        duration: Processing duration in seconds
        output_path: Output file path
    """
    total = len(results)
    yes_count = sum(1 for r in results if r[0] == "Yes")
    no_count = sum(1 for r in results if r[0] == "No")
    error_count = sum(1 for r in results if r[0] == "Error")

    logging.info("=" * 60)
    logging.info("Processing complete!")
    logging.info("=" * 60)
    logging.info(f"Total reports processed: {total}")
    logging.info(f"Imaging-related addenda found: {yes_count} ({yes_count/total*100:.1f}%)")
    logging.info(f"No imaging-related addenda: {no_count} ({no_count/total*100:.1f}%)")
    logging.info(f"Errors: {error_count} ({error_count/total*100:.1f}%)")
    logging.info(f"Processing time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logging.info(f"Average time per report: {duration/total:.1f} seconds")
    logging.info(f"Output file: {output_path}")

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print('=' * 60)
    print(f"Total reports: {total}")
    print(f"Imaging-related: {yes_count} ({yes_count/total*100:.1f}%)")
    print(f"Not imaging-related: {no_count} ({no_count/total*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/total*100:.1f}%)")
    print(f"Time: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"\nOutput: {output_path}\n")


def print_summary_production(results: List[Tuple[str, str, str]], duration: float,
                            output_path: str, processed_count: int, total_count: int):
    """Print summary for production mode.

    Args:
        results: Processing results for this batch
        duration: Processing duration in seconds
        output_path: Output file path
        processed_count: Total processed so far
        total_count: Total rows in dataset
    """
    batch_total = len(results)
    yes_count = sum(1 for r in results if r[0] == "Yes")
    no_count = sum(1 for r in results if r[0] == "No")
    error_count = sum(1 for r in results if r[0] == "Error")

    logging.info("=" * 60)
    logging.info("Batch processing complete!")
    logging.info("=" * 60)
    logging.info(f"This batch: {batch_total} reports")
    logging.info(f"  Imaging-related: {yes_count}")
    logging.info(f"  Not imaging-related: {no_count}")
    logging.info(f"  Errors: {error_count}")
    logging.info(f"Progress: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%)")
    logging.info(f"Remaining: {total_count - processed_count} reports")
    logging.info(f"Batch time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logging.info(f"Average time per report: {duration/batch_total:.1f} seconds")
    logging.info(f"Output file: {output_path}")

    print(f"\n{'=' * 60}")
    print("Batch processing complete!")
    print('=' * 60)
    print(f"This batch: {batch_total} reports")
    print(f"  Imaging-related: {yes_count}")
    print(f"  Not imaging-related: {no_count}")
    print(f"  Errors: {error_count}")
    print()
    print(f"Overall progress: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%)")
    print(f"Remaining: {total_count - processed_count} reports")
    print(f"Time: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"\nOutput: {output_path}")

    if processed_count < total_count:
        print(f"\n💡 Run again to continue processing (batch size: {config.BATCH_SIZE or 'all remaining'})\n")
    else:
        print(f"\n✓ All {total_count} reports have been processed!\n")


def main():
    """Main execution function."""
    # Setup logging
    setup_logging(config.RUN_MODE)

    logging.info("=" * 60)
    logging.info("Starting radiology report processing")
    logging.info("=" * 60)
    logging.info(f"Input CSV: {config.INPUT_CSV}")
    logging.info(f"Output directory: {config.OUTPUT_DIR}")
    logging.info(f"Model: {config.MODEL_NAME}")
    logging.info(f"Reasoning effort: {config.REASONING_EFFORT}")

    # Initialize Ollama client
    logging.info("Initializing Ollama client...")
    client = OllamaClient()

    # Check Ollama availability
    if not client.check_availability():
        error_msg = f"Ollama is not running or model '{config.MODEL_NAME}' is not available. " \
                    f"Please ensure Ollama is running and the model is installed."
        logging.error(error_msg)
        print(f"\n❌ {error_msg}\n")
        sys.exit(1)

    logging.info(f"✓ Ollama is running with model '{config.MODEL_NAME}'")
    print(f"✓ Ollama is running with model '{config.MODEL_NAME}'")

    # Read CSV
    try:
        header, all_rows = read_all_csv_rows(config.INPUT_CSV)
    except FileNotFoundError as e:
        logging.error(str(e))
        print(f"\n❌ {e}\n")
        sys.exit(1)

    # Run in appropriate mode
    if config.RUN_MODE == "test":
        run_test_mode(client, header, all_rows)
    elif config.RUN_MODE == "production":
        run_production_mode(client, header, all_rows)
    else:
        logging.error(f"Invalid RUN_MODE: {config.RUN_MODE}. Must be 'test' or 'production'")
        print(f"\n❌ Invalid RUN_MODE in config.py: {config.RUN_MODE}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
