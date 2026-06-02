# PR007 NCU Summary (All Cases)

Generated: 2026-06-02T12:08:29.811817+00:00

| Case | NCU status | DRAM % | SM % | Tensor % | Warps % | Duration (us) |
|------|------------|--------|------|----------|---------|---------------|
| Baseline FWT (14 butterfly stages, batch=1) | OK | 33.4 | 10.3 | — | 21.9 | 3.17 |
| TC: phase split | OK | 0.4 | 67.3 | — | 37.5 | 12.54 |
| TC: modulo precompute | OK | 18.4 | 17.0 | — | 55.4 | 2.91 |
| Ozaki MMA grid (OZAKI_NCU_PROFILE=1) | LaunchFailed | — | — | — | — | — |
| Ozaki MMA persistent (production) | LaunchFailed | — | — | — | — | — |

## Detail

See `ncu_summary.txt`.
