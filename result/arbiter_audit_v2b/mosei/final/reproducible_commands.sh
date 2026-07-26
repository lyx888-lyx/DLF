# Stage23A-v2b reproducible commands
# Executed from /code/DLF-mosei-arbiter-audit-v1
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_authorize.py
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_local_competence.py --phase develop
# Freeze/commit/push selection before outer.
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_local_competence.py --phase outer
# Serialization-only recovery from saved aggregate metrics; no outer reload.
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_local_competence.py --phase finalize-saved
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_finalize.py
