# Track 4 Observer-Slice Transport Probe - 2026-05-29

## What Changed

Track 4 now has a small observer-atlas diagnostic in `core/observer_slice_transport.py`.

Instead of asking only whether a walker can traverse one static terrain, the diagnostic asks whether semantic movement and observer switching commute:

```text
semantic first:
  (article A, observer X) -> (article B, observer X) -> (article B, observer Y)

observer first:
  (article A, observer X) -> (article A, observer Y) -> (article B, observer Y)
```

The absolute action gap between those two routes is recorded as `holonomy_action`.

## Property/Theft Toy Result

Persistent artifact:

`outputs/track4_observer_slice_transport_probe/property_theft_holonomy_20260529/observer_slice_transport_summary.json`

Summary:

- `record_count`: `4`
- `mean_holonomy_action`: `1.5`
- `mean_null_holonomy_action`: `0.0`
- `mean_excess_holonomy_action`: `1.5`

Interpretation:

- Translation-only observer slices produce zero holonomy.
- Warped observer slices produce positive holonomy when `property -> theft` is direct in one chart but barrier-like in another.
- This supports the engineering reframing of Track 4 as observer transport / semantic holonomy rather than only MCMC-style traversal on one chart.

## Verification

Passed:

```powershell
python -m pytest --basetemp="$env:TEMP\btv3_slice_transport_targeted2" test_track4_action_graph.py test_observer_manifold_bundle.py test_observer_recenter_meaning_probe.py test_observer_recenter_robustness_suite.py -q
```

Passed:

```powershell
$tests = Get-ChildItem -Path . -File | Where-Object { $_.Name -like 'test_track4_*.py' -or $_.Name -like 'test_observer_*.py' } | ForEach-Object { $_.Name }; python -m pytest --basetemp="$env:TEMP\btv3_slice_transport_track4_observer2" @tests -q
```

The new tests include:

- Translation-only observer slices have zero holonomy.
- Property/theft-style warped observer charts have positive excess holonomy over translation null.
- Existing observer manifold bundle coordinates can be passed into the observer-slice transport diagnostic.
