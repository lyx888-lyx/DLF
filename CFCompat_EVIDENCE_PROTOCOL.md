# CFCompatKD Distillation Evidence Closure Protocol

## Scope

Stage 18 evaluates the frozen CFCompatKD method; it does not tune or extend it.
The student, clean Teacher, evaluator, compatibility definition, prediction KD
loss, missing-mode schedule, supervised losses, optimizer family, data splits,
and official metric implementation are frozen.

The only registered interventions are:

- no KD (ModDrop);
- Uniform KD;
- Equal-Mass Global KD;
- Mode-Mean KD;
- deterministic within-mode Shuffled-Gate KD;
- deterministic within-mode Shuffled-Teacher KD;
- train-label-only Oracle-Direction KD;
- the existing CFCompatKD gate;
- unified student versus fixed-mode label-supervised specialist.

MOSI Test remains locked through Stages 18A--18F. Historical aggregate Test
metrics may be mentioned as background but may not select a method, epoch, gate,
or control. Every run must declare all Test-access flags false and a locked
access count of zero.

## Ordered gates

1. Recover the historical no-Test Online trainer on seed 1114.
2. Audit unified-versus-specialist learnability on seeds 1114 and 1111.
3. Run all eight frozen KD controls on seed 1114.
4. Repeat all eight controls on seed 1111 without changing definitions.
5. Diagnose transfer, harmful imitation, Teacher benefit, and compatibility.
6. Apply the preregistered two-seed continuation gate.
7. Run five-seed validation evidence only if that gate passes.
8. Unlock one-time Test evaluation only if the five-seed criteria pass.

Negative results do not cancel the registered two-seed mechanism analyses.
