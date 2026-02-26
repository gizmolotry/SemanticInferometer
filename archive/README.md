# Archive Folder Contract

This folder stores legacy snapshots and historical artifacts kept for reference.

## Current contents
- `backup_nov24/`: older backup snapshot moved from repo root
- `ancien/outputs/`: legacy output tree moved from `ancien/outputs`

## Rules
- Archive is read-only by convention unless explicitly doing a migration.
- Do not write fresh experiment outputs here.
- Keep historical folder names and internal structure unchanged when possible.
- If new historical material is archived, add a short note in this file with date and source path.
