# ADPEP cross-dataset freeze protocol

MOSI is used only for the initial mechanism verification. Once the Stage 9B
MOSI run is complete, the ADPEP-All implementation and projection rules are
frozen.

Future MOSEI execution must use:

1. the same implementation and float32 evaluator-decision wrapper;
2. the same fixed seed set 1111–1115;
3. the same anchor rule: minimum MOSEI validation J, lower seed on ties;
4. the same ADPEP-All main method and ADPEP-57 ablation status;
5. the same label-free prediction-freezing order;
6. the same sample-binding, SHA, decision-preservation, and fallback gates;
7. the same metrics, MissingMacro, J, and retention formulas.

The command interface already accepts `--dataset mosei`. Dataset paths are
derived from the dataset argument. Neither the projection interval nor anchor
selection contains a MOSI seed, MOSI sample count, MOSI label distribution, or
MOSI test rule.

MOSEI may not be used to redesign the projection, change a boundary, relax
Acc2/F1 preservation, choose ADPEP-57 as the main method, tune ensemble weights,
or fit a calibrator. Its anchor may be selected only from MOSEI validation J.
MOSEI test labels and metrics may be read only after the frozen projected
predictions have been written and hashed.

No MOSEI training, inference, or evaluation is part of Stage 9B.
