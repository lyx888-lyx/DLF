# CFCompatKD Locked Test-Once Protocol

MOSI Test is locked until the preregistered Stage 18F five-seed validation
criteria all pass. If they do not pass, no unlock manifest may be produced and
the locked access count must remain zero.

After an eligible unlock, the frozen methods are ModDrop, Uniform KD,
Equal-Mass Global KD, Shuffled-Gate KD, Oracle-Direction KD, and CFCompatKD.
Oracle labels are used only to build the train-time KD gate.

The one allowed Test evaluation must:

1. record the frozen branch, commit, code hashes, checkpoints, and method list;
2. generate label-free Test predictions;
3. verify sample order and bindings;
4. freeze prediction SHA-256 values;
5. read Test labels only after prediction freezing;
6. compute the official metrics exactly once;
7. record `locked_test_access_count=1`;
8. prohibit method, checkpoint, gate, or analysis changes and prohibit reruns.
