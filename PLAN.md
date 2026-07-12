# naviFyTools Improvement Plan

Agreed scope from the 2026-07-12 audit (three-agent sweep of naviDJ.py, app.py/UI, portLibrary.py/portGenres.py). Any session on any machine can pick up from here.

## Owner decisions (constraints for all work)

- **No playlist ordering/flow logic.** Playlists are played shuffled; selection quality is what matters.
- **Artist cap must be adaptive.** Sometimes a 1–2 artist mix is the goal. Cap songs-per-artist only when the prompt names no artists; when `explicit_artists` is non-empty, no cap.
- **Variant dedup**: collapse duplicate variants preferring the remaster over the original. Live versions are *distinct* tracks and may coexist with studio versions.
- **Era matching**: loose scoring bonus only, never a filter.
- **Genre data**: keep ALL MusicBrainz tags in portGenres (moods/eras like "80s", "chill") — no whitelist filtering. Owner reruns the genres script manually after.
- **Embeddings**: include genre/album/year in song embedding text (invalidates `.pkl` caches — one-time rebuild).
- **Deferred / out of scope**: cross-run repetition memory, Last.fm, Spotify popularity (revisit later), Spotify release dates (keep dropping), duration/bpm/moods/userRating.

## Workstream A — Post-selection quality pass (naviDJ.py)

1. Adaptive artist cap (post-selection, only when no explicit artists; replace over-cap picks with best-scoring alternatives).
2. Fix bug: `explicit_artists` is computed and threaded through but never included in the LLM prompt.
3. Variant dedup of candidates pre-LLM + final safety pass. Normalized-title collapse; prefer starred > remaster > most-played. Live-ness is part of the dedup key (live ≠ studio).
4. Relevance-weighted padding: `ensure_min_songs` and chunk-failure fallback pick by relevance score, not random.

## Workstream B — Data enrichment

1. Read full Subsonic `genres` array (currently only first `genre` string) — use in scoring and LLM view.
2. portGenres.py: keep all MusicBrainz tags (remove whitelist filter).
3. Loose era bonus from `releaseYear` when prompt implies an era.
4. Embedding text: title + artist + genres + album + year.

## Workstream C — Web UI (Flask + vanilla JS)

1. naviDJ emits structured `PLAYLIST_JSON:` final line; frontend renders the tracklist.
2. Structured progress view instead of raw log dump (raw log behind a toggle).
3. UX: double-submit guard, cancel, prompt history (localStorage), per-run model override + echo of model/config used.
4. `run_dj` error handling — surface subprocess launch failures instead of silent dead stream.

## Workstream D — Bug & reliability sweep

1. **portLibrary.py data loss (urgent)**: updating an existing playlist uses `createPlaylist`, which replaces the song list — unmatched tracks get silently deleted. Use non-destructive update semantics.
2. portLibrary.py: `songCount` on Subsonic search (default 20 → false "no match"); verify Spotify `items[0]` before accepting; HTTP timeouts + error checks (auth failure ≠ "no match"); cache MusicBrainz lookups.
3. naviDJ.py: `fetch_starred_ids()` out of pagination loop; retry/backoff for OpenAI-mode transient errors.
4. Dead code: `_metadata_score` wrong-key debug (real field `_relevance_score`), unused `extract_prompt_artists`, unreachable fuzzy-match net in `_sanitize_playlist`, uncalled `filter_genres`.
5. Security (LAN-appropriate): don't return API keys from `/get_config` or embed secrets in HTML; disable Flask `debug=True`.

## Execution order

A → D → B → C, committed per workstream. (portLibrary part of D can run in parallel with A — disjoint files.)
