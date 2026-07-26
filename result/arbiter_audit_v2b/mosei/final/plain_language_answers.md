# Stage23A-v2b plain-language answers

1. **Is Expert performance more regular near similar history?** Yes, but only
   weakly. Outer local-risk/error Spearman is
   0.196243 (A) and
   0.247474 (B).
2. **Best neighborhood:** Hybrid in both directions. A selected
   H/cosine/K30/beta=0.25; B selected H/standardized-Euclidean/K60/beta=0.50.
3. **True versus random/shuffled:** true retrieval is slightly better in final
   J, but the mean margin is only -0.000269, below the frozen -0.002 gate.
4. **Best DRS:** Local-DW. Local-DS is clearly worse and DWS does not improve
   upon DW on outer.
5. **Safe fallback:** not reliably. The frozen safe rule changes J by
   -0.000086 (A) and -0.000175 (B), while top-10% triggered true gain
   is -0.000844 (A) and
   +0.000652 (B).
6. **Direction agreement:** local-risk correlations are positive and safe
   Delta J is negative in both directions, but high-confidence gain is not
   direction-consistent.
7. **Promotion:** no. Final status is `LOCAL_COMPETENCE_WEAK`.
8. **Why not pass:** neighborhoods are not sparse or source-monopolized; every
   query receives the frozen K with zero same-source and duplicate-source
   neighbors. The limiting factor is weak competence-to-gain precision and
   poor beneficiary ranking, with a modest outer density shift.
9. **Continue post-hoc Judge?** There is no evidence to promote a formal neural
   Judge or Student. Because the frozen conclusion is WEAK rather than FAIL,
   permanent route closure is not automatically asserted; both remain locked
   pending an explicit research decision.
