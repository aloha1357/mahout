import csv
import os


def parse_ncu_csv(file_path):
    metrics = {}
    if not os.path.exists(file_path):
        return None
    
    content = ""
    for encoding in ['utf-8', 'utf-16', 'cp950']:
        try:
            with open(file_path, encoding=encoding) as f:
                content = f.read()
            break
        except Exception:
            continue
    else:
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
    lines = content.splitlines()
    csv_start = 0
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith('\ufeff"ID"'):
            csv_start = i
            break
    else:
        return None
    
    import io
    reader = csv.DictReader(io.StringIO("\n".join(lines[csv_start:])))
    for row in reader:
        metric_name = row.get('Metric Name')
        metric_value = row.get('Metric Value')
        if not metric_name:
            continue
        if metric_name not in metrics:
            metrics[metric_name] = []
        metrics[metric_name].append(metric_value)
    
    # Average numeric values
    summary = {}
    for name, values in metrics.items():
        try:
            # Remove commas from values like "1,024"
            numeric_values = [float(v.replace(',', '')) for v in values if v not in ['nan', 'n/a', '']]
            if numeric_values:
                summary[name] = sum(numeric_values) / len(numeric_values)
            else:
                summary[name] = "N/A"
        except (ValueError, TypeError):
            summary[name] = values[0] if values else "N/A"
            
    return summary

def generate_summary():
    files = {
        "Baseline FWT": "qdp/qdp-kernels/reports/ncu_fwt_baseline.csv",
        "Phase Split": "qdp/qdp-kernels/reports/ncu_phase_split.csv",
        "Modulo Precompute": "qdp/qdp-kernels/reports/ncu_modulo_precompute.csv",
        "Ozaki Grid (Attempt)": "qdp/qdp-kernels/reports/ncu_ozaki_grid.csv"
    }
    
    with open("AdaptiveGEMM_repo/reports/ncu_summary_new.txt", "w") as out:
        out.write("PR007 NCU Summary — Regenerated 2026-06-02\n")
        out.write("GPU: RTX 4060 (SM 8.9, WDDM) | NCU 2024.3.0.0 | CUDA 13.0\n\n")
        
        for label, path in files.items():
            out.write(f"{label}\n" + "-"*len(label) + "\n")
            data = parse_ncu_csv(path)
            if data:
                out.write(f"  DRAM throughput: {data.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')}\n")
                out.write(f"  L2 throughput: {data.get('lts__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')}\n")
                out.write(f"  SM throughput: {data.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')}\n")
                out.write(f"  Warps active: {data.get('smsp__warps_active.avg.pct_of_peak_sustained_active', 'N/A')}\n")
                out.write(f"  Registers / thread: {data.get('launch__registers_per_thread', 'N/A')}\n")
                out.write(f"  Shared memory / block: {data.get('launch__shared_mem_per_block', 'N/A')}\n")
                out.write(f"  Kernel duration (ns): {data.get('gpu__time_duration.sum', 'N/A')}\n")
                if 'sm__pipe_tensor_active.avg.pct_of_peak_sustained_active' in data:
                    out.write(f"  Tensor Core active: {data.get('sm__pipe_tensor_active.avg.pct_of_peak_sustained_active', 'N/A')}\n")
            else:
                out.write("  Data not available or profiling failed.\n")
            out.write("\n")

if __name__ == "__main__":
    generate_summary()
