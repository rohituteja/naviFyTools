"""Plain-assert self-checks for the naviDJ length-trim and fuzzy-artist fixes.

Run:  ./pipenv/bin/python test_naviDJ.py
No framework, no fixtures -- matches this repo's minimal style.
"""

from naviDJ import (
    extract_prompt_entities,
    _trim_to_relevance,
    _context_overlaps_explicit_artists,
)


def test_fuzzy_matches_typod_artist_only():
    """Typo'd artist name fuzzy-matches the real one, not an unrelated decoy."""
    artists = ["Michael Jackson", "Janet Jackson"]
    ents = extract_prompt_entities("micheal jackson mix", artists, [], [])
    assert ents["artists"] == ["Michael Jackson"], ents["artists"]


def test_vibe_only_prompt_finds_no_artist():
    """No artist named in the prompt -> fuzzy path must not false-positive."""
    artists = ["Michael Jackson", "Janet Jackson", "Daft Punk"]
    ents = extract_prompt_entities("funky groovy", artists, [], [])
    assert ents["artists"] == [], ents["artists"]


def test_correct_spelling_uses_exact_path():
    """Correctly spelled artist uses exact/partial match; fuzzy never fires."""
    artists = ["Logic", "Childish Gambino", "Janet Jackson"]
    ents = extract_prompt_entities("logic and gambino", artists, [], [])
    assert ents["artists"] == ["Logic"], ents["artists"]


def test_trim_keeps_highest_relevance():
    """Trimming an oversized playlist keeps the highest-scoring picks."""
    filtered = [
        {"id": "a", "title": "A", "_relevance_score": 1},
        {"id": "b", "title": "B", "_relevance_score": 9},
        {"id": "c", "title": "C", "_relevance_score": 5},
        {"id": "d", "title": "D", "_relevance_score": 7},
        {"id": "e", "title": "E", "_relevance_score": 3},
    ]
    picks = [{"id": s["id"], "title": s["title"]} for s in filtered]  # 5 picks
    trimmed = _trim_to_relevance(picks, filtered, 3)
    assert [p["id"] for p in trimmed] == ["b", "d", "c"], trimmed


def test_trim_noop_and_missing_score():
    """Already at/under min_songs is a no-op; unknown id defaults to score 0."""
    filtered = [{"id": "a", "_relevance_score": 5}]
    picks = [{"id": "a", "title": "A"}, {"id": "ghost", "title": "G"}]
    assert _trim_to_relevance(list(picks), filtered, 5) == picks  # no-op
    assert _trim_to_relevance(list(picks), filtered, 1) == [{"id": "a", "title": "A"}]


def test_context_discarded_when_no_explicit_artist_overlap():
    """A context playlist sharing none of the requested artists gets discarded."""
    context = [
        {"artist": "The Strokes"},
        {"artist": "Djo"},
        {"artist": "The Voidz & Julian Casablancas"},
    ]
    assert not _context_overlaps_explicit_artists(context, ["Michael Jackson"])


def test_context_kept_when_explicit_artist_present():
    """Overlap (even from a multi-artist credit) keeps the context playlist."""
    context = [
        {"artist": "The Strokes"},
        {"artist": "Michael Jackson & Paul McCartney"},
    ]
    assert _context_overlaps_explicit_artists(context, ["Michael Jackson"])


if __name__ == "__main__":
    for fn in [
        test_fuzzy_matches_typod_artist_only,
        test_vibe_only_prompt_finds_no_artist,
        test_correct_spelling_uses_exact_path,
        test_trim_keeps_highest_relevance,
        test_trim_noop_and_missing_score,
        test_context_discarded_when_no_explicit_artist_overlap,
        test_context_kept_when_explicit_artist_present,
    ]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("all self-checks passed")
