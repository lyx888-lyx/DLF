# MCAO Loss Dependency Specification

The actual CFCompatKD objective has two student views per batch. The full LAV view receives the complete DLF objective. The sampled missing view receives five task heads plus compatibility-gated prediction KD.

The sampled missing-view task helper has no presence argument. Therefore `missing_audio_specific_task` and `missing_visual_specific_task` remain in the scalar total even when their source modality is absent. Missing-view reconstruction, consistency, orthogonality, and triplet losses are not computed at all; their full-LAV counterparts remain semantically valid.

## Statically inconsistent losses

- `missing_audio_specific_task`: invalid-but-computed modes LV, L
- `missing_visual_specific_task`: invalid-but-computed modes LA, L

The CSV is the authoritative row-level dependency graph:
`result/missing_baseline/mcao_v1/mosi/stage17a_loss_audit/stage17_loss_dependency_graph.csv`.
