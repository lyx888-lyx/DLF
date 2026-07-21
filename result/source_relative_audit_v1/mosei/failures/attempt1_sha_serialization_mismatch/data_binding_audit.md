# Stage 21A data binding audit

- Status: `STAGE21A_BLOCKED_ID_BINDING_ERROR`
- Sample ID format: `video_id$_$clip_id`
- `video_id` denotes the original source video; no speaker metadata is present.
- Train/Valid source overlap: **0**
- Train/Valid sample overlap: **0**
- Locked Test access count: **0**

| Split | Samples | Sources | Clips/source mean | median | p90 | max | Order/cache bound |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 16326 | 2249 | 7.2592 | 5.0 | 15.0 | 98 | False |
| valid | 1871 | 300 | 6.2367 | 4.5 | 13.0 | 39 | False |
