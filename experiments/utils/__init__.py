"""
Shared utilities for experiments.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def save_results(
    results: List[Dict],
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save experiment results to JSON with metadata.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
        metadata: Optional metadata to include
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_results': len(results),
        'metadata': metadata or {},
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {output_file}")


def results_to_markdown(
    results: List[Dict],
    title: str,
    columns: List[str],
    key_mapping: Optional[Dict[str, str]] = None,
    format_spec: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convert results to markdown table.
    
    Args:
        results: List of result dictionaries
        title: Table title
        columns: Column headers
        key_mapping: Map column names to result dict keys
        format_spec: Format specifications for values (e.g., {'R1': '.4f'})
    
    Returns:
        Markdown string
    """
    if key_mapping is None:
        key_mapping = {col: col.lower().replace(' ', '_') for col in columns}
    if format_spec is None:
        format_spec = {}
    
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    
    for r in results:
        row_values = []
        for col in columns:
            key = key_mapping.get(col, col.lower())
            val = r.get(key, '-')
            
            # Apply format spec
            if col in format_spec and val != '-' and val is not None:
                try:
                    val = f"{val:{format_spec[col]}}"
                except (ValueError, TypeError):
                    pass
            
            row_values.append(str(val))
        
        lines.append("| " + " | ".join(row_values) + " |")
    
    lines.append("")
    lines.append(f"*Generated: {datetime.now().isoformat()}*")
    
    return "\n".join(lines)


def print_summary_table(
    results: List[Dict],
    columns: List[str] = None,
    key_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """
    Print a summary table to console.
    
    Args:
        results: List of result dictionaries
        columns: Column headers (default: common metrics)
        key_mapping: Map column names to result dict keys
    """
    if columns is None:
        columns = ['Variant', 'Sel%', 'Spec%', 'R@1', 'ΔR@1', 'Flip%', 'A/B', 'NF/1k', 'Purity']
    
    if key_mapping is None:
        key_mapping = {
            'Variant': 'variant',
            'Sel%': 'sel_pct',
            'Spec%': 'spec_pct',
            'R@1': 'r1',
            'ΔR@1': 'delta_r1',
            'Flip%': 'flip_pct',
            'A/B': 'A/B',
            'NF/1k': 'net_flip_per_1k',
            'Purity': 'purity',
        }
    
    # Compute column widths
    col_widths = {}
    for col in columns:
        key = key_mapping.get(col, col.lower())
        max_len = len(col)
        for r in results:
            val = r.get(key, '-')
            if val is None:
                val = '-'
            elif isinstance(val, float):
                if 'R@1' in col or 'ΔR@1' in col:
                    val = f"{val:.4f}"
                elif '%' in col:
                    val = f"{val:.1f}%"
                elif 'NF' in col:
                    val = f"{val:+.1f}"
                else:
                    val = f"{val:.2f}"
            max_len = max(max_len, len(str(val)))
        col_widths[col] = max_len + 2
    
    # Print header
    header = "".join(f"{col:^{col_widths[col]}}" for col in columns)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    # Print rows
    for r in results:
        row_parts = []
        for col in columns:
            key = key_mapping.get(col, col.lower())
            val = r.get(key, '-')
            if val is None:
                val = '-'
            elif isinstance(val, float):
                if 'R@1' in col or 'ΔR@1' in col:
                    val = f"{val:.4f}"
                elif '%' in col:
                    val = f"{val:.1f}%"
                elif 'NF' in col:
                    val = f"{val:+.1f}"
                else:
                    val = f"{val:.2f}"
            row_parts.append(f"{str(val):^{col_widths[col]}}")
        print("".join(row_parts))
    
    print("=" * len(header))
