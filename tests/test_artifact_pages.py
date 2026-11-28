"""Artifacts drawn as one finished page: charts to scale, agent HTML kept static, one renderer."""

import pytest

from opentrain import artifact_page as ap
from opentrain import artifacts as art

TEXT = [{"type": "text", "text": "x"}]


def test_ticks_are_round_and_cover_the_range():
    t, step = ap.ticks(0, 0.93)
    assert t[0] == 0 and t[-1] >= 0.93 and step in (0.2, 0.25)
    t, step = ap.ticks(3, 1206)
    assert t[0] == 0 and t[-1] >= 1206 and step in (200, 250, 500)


def test_a_line_chart_is_drawn_to_scale_in_the_page_colours():
    b = {"type": "chart", "kind": "line", "title": "Training loss", "x_label": "step",
         "series": [{"name": "loss", "points": [[0, 2.0], [10, 1.0], [20, 0.5]]}]}
    svg = ap.chart_svg(b)
    assert svg.startswith("<figure>") and "<polyline" in svg and "var(--c1)" in svg
    assert ">0.5<" in svg  # the last value is labelled
    assert "#" not in svg.split("<svg", 1)[1].split("</svg>")[0].replace("&#", "")  # no fixed colours


def test_a_bar_chart_groups_series_by_category():
    b = {"type": "chart", "kind": "bar", "unit": "%", "max": 100,
         "series": [{"name": "base", "points": [["css", 30], ["api", 0]]},
                    {"name": "trained", "points": [["css", 60], ["api", 10]]}]}
    svg = ap.chart_svg(b)
    assert svg.count("<rect") == 4 and ">100%<" in svg and 'class="legend"' in svg


def test_agent_html_stays_static():
    dirty = ('<div class="tl" onclick="steal()"><script>fetch("https://x.example")</script>'
             '<a href="javascript:alert(1)">x</a><iframe src="https://x.example"></iframe>'
             '<style>.tl{color:var(--ink)}</style><svg viewBox="0 0 10 10"><rect width="5" height="5" fill="var(--c1)"/></svg></div>')
    clean = ap.clean_html(dirty)
    for bad in ("<script", "onclick", "javascript:", "<iframe"):
        assert bad not in clean
    assert "<style>" in clean and "<svg" in clean


def test_new_blocks_are_checked(tmp_path):
    ok = [{"type": "callout", "tone": "warn", "title": "Few tasks", "text": "6 held-out tasks."},
          {"type": "chart", "kind": "line", "series": [{"name": "loss", "points": [[0, 1], [1, 0.5]]}]},
          {"type": "html", "html": "<p>hi</p><script>x()</script>"}]
    a = art.save(tmp_path, kind="run_page", title="Run", blocks=ok)
    assert "<script" not in a["blocks"][2]["html"]
    for bad in ({"type": "callout", "tone": "loud", "text": "x"},
                {"type": "chart", "kind": "pie", "series": [{"points": [[0, 1]]}]},
                {"type": "chart", "series": [{"points": [1, 2]}]},
                {"type": "html", "html": "x" * (art.MAX_HTML + 1)}):
        with pytest.raises(art.ArtifactError):
            art.save(tmp_path, kind="run_page", title="Run", blocks=[bad])


def test_the_page_is_the_same_in_the_app_and_shared(tmp_path):
    a = art.save(tmp_path, kind="run_page", title="Run <1>", blocks=TEXT,
                 decision={"question": "Go?", "options": ["yes", "no"]})
    shared, inside = art.to_html(a), art.to_html(a, embed=True, theme="dark")
    assert "Run &lt;1&gt;" in shared and "Go?" in shared  # the shared page carries the decision
    assert "Go?" not in inside and 'data-theme="dark"' in inside  # the app draws the decision itself


def test_the_page_route_forbids_scripts_and_outside_loads(live):
    import urllib.request

    c = live.client(live.proj)
    c.get("/api/status")  # opens the project, so its state folder exists
    art.save(live.proj / ".opentrain", kind="run_page", title="Run", blocks=TEXT)
    req = urllib.request.Request(c._url("/api/artifacts/run-page-1/html", {"embed": "1"}), headers=c.headers)
    with urllib.request.urlopen(req) as r:
        csp, body = r.headers["Content-Security-Policy"], r.read().decode()
    assert "default-src 'none'" in csp and "script-src" not in csp and "frame-ancestors 'self'" in csp
    assert 'class="embed"' in body
