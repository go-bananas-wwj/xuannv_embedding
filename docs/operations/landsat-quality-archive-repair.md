# Restore archived quality flags before final annual-data claims

Input-only inspection, independent of downstream scores, found that the legacy
China Landsat rasters store fractional QA values (for example 0.40016) instead of
integer QA_PIXEL flags. This value corresponds to applying the reflectance scale
and offset to the actual QA code 21824. The current annual reader correctly
rejects those fractional flags; consequently its reported Landsat coverage is zero
in all twelve months in both China regions. The annual screen therefore changed
source availability as well as temporal coverage. Its negative result cannot
isolate a pure effect of adding seasons.

Original NetCDF observations retain raw reflectance DN and integer QA_PIXEL. Use
them to regenerate only the affected Landsat observations in a new directory:
scale the six reflectance bands once, preserve QA bits, and reproject QA with
nearest-neighbor resampling. Do not guess cloud flags by truncating fractional
values, overwrite archived inputs, or change arrays read by the active matched
training jobs. Keep the existing trial as a version using its registered available
observations; do not present it as complete three-source annual input.

An actual reprocessing pilot exposed a second problem in the current preparation
routine: its validity mask is uint8, and using it in a quality-array index selects
integer rows rather than valid pixels. This leaves nodata quality values at pixels
marked valid. A daily NetCDF regression with multiple rows and a cloudy/clear
scene pair reproduces this for both S2 SCL and Landsat QA. Convert the assignment
mask to Boolean explicitly; verify every output quality pixel and the first-valid
scene rule. This repair affects future materialization, not existing trained
weights or frozen embedding arrays.

Before bulk regeneration, compare newly written integer quality samples directly
with the original NetCDF at corresponding coordinates. An initial pilot date has
no observations at the chosen patch, which must remain missing; use an actually
observed archive date for the positive sample gate. Preserve those empty outputs
as evidence that the filter does not invent coverage.

Record source/checkpoint identities, annual coverage and all failed comparisons.
The matched model fits currently in progress retain their original data hashes.
Any corrected-input model comparison is a separately registered shared wave with
all three regions, genuine no-imagery controls, unchanged primary tasks/masks and
validation-based selection. It must not overwrite or silently relabel earlier
results. No manuscript victory claim follows this repair.
