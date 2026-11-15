---
version: 1
slug: "web-app-pages-index-vue"
primary_target: "web/app/pages/index.vue"
related_targets: []
---

# Surface brief: analyst workspace (chat + answer + trace)

Scope: the main dashboard screen — thread rail, chat with streamed answers and charts, run-trace panel, background-task tray, steer/queue input. Visitor mode: Operate.
Audience and job: a senior AI engineer judging the agent (and the owner operating it); ask, watch the agent work, inspect proof and cost.
Constraints: Nuxt + Vue 3 TS client-side; light and night themes both first-class (owner, 2026-11-15); chart performance budgets per spec §8; synthetic demo values labelled illustrative.
Memorable moment: the forecast fan drawn as labelled contour lines, continuing in survey red from black-ink history at a marked origin.
Unresolved: character name; exact night-mode tokens (resolved in build, documented in DESIGN.md at finish).

## Direction contract

THESIS: Uncertainty is terrain. Every range the agent reports is drawn as contour lines with index labels, and the run itself reads as a surveyed profile. Refuses the category default of a dark neon AI dashboard with glowing cards.
OWN-WORLD: USGS quadrangle palette — near-white survey paper, black ink, contour brown, water blue, woodland green tint, survey red reserved only for agent-predicted marks. Night mode is the same sheet under a red-safe map light: deep blue-black ground, warm paper-white ink, contours in lifted brown, red still reserved for predictions. Public Sans with tabular figures; 1 device-pixel hairlines; verification states as printed survey marks (benchmark triangle), never pills; graticule ticks on panel edges; one visible module grid.
STORY: The visitor asks; the swarm character works; the answer arrives with a traced verification mark, a chart that shows history, forecast, ranges, last year and the origin, and a backtest scorecard; the trace shows tokens, cache, cost, context budget remaining, step durations and reasoning. Background tasks run in the rail tray while they keep asking or steer.
FIRST VIEWPORT: 48px top bar (name, dataset and cutoff, model). 240px thread rail with background-task tray at its foot. Centre chat (question right, answer left: swarm + mark, narrative with red predicted values, forecast figure, scorecard), steer/queue input pinned at bottom. 420px trace panel with ticking counters, context budget bar, run profile bars with right-aligned durations, reasoning summary.
FORM: Survey Sheet, assigned candidate 5 of 7 on the grounded list; seed:28d2fe6d9cdfd1a3
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
