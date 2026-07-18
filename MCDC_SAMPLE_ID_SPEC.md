# MCDC MOSI Sample ID Specification

Observed examples: train: `03bSnISJMiM$_$11`, train: `03bSnISJMiM$_$10`, train: `03bSnISJMiM$_$13`, valid: `WKA5OygbEKI$_$20`, valid: `WKA5OygbEKI$_$21`, valid: `WKA5OygbEKI$_$22`.

The exact parser is `^(?P<video_id>.+?)\\$_\\$(?P<segment_index>[0-9]+)$`. `video_id` is the non-empty prefix before the final `$_$` delimiter and `segment_index` is the trailing non-negative integer. IDs that do not match raise an error. Duplicate `(split, video_id, segment_index)` bindings are fatal. Context is reconstructed by exact integer offsets `i-3`, `i-2`, `i-1`; missing offsets remain left padding and are never filled from another sample.
