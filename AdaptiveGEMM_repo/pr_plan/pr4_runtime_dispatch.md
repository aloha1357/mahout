# PR4: Runtime Dispatch & Hardware Fallbacks

## Scope
Ensure cross-platform stability by dynamically detecting GPU architectures and capabilities.

## Target Changes
- Architecture check logic (SM75 vs SM80+)
- TensorCore experimental paths (opt-in)
- Fallback paths for WDDM and unsupported architectures

## Review Focus
- Robustness of capability probing
- Graceful degradation on older hardware
- Safety of experimental feature flags

## Merge Blockers
- Any crashes on CI (which uses older/limited runners)
- Unhandled `LaunchFailed` exceptions
