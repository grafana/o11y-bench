import copy
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from grading.dashboard_diff import diff_dashboards, render_compact_diff, seed_dashboard
from grading.env_context import VerifierContext
from grading.facts import FactResult, dump_truth, load_truth
from grading.judge import build_judge_criteria
from grading.models import Problem

PROMETHEUS = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}
UNCHANGED = "no changes on any compared surface"


def _panel(panel_id: int, title: str, /, **extra: Any) -> dict[str, Any]:
    return {
        "id": panel_id,
        "title": title,
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8 * panel_id},
        "datasource": PROMETHEUS,
        "targets": [{"expr": "up", "refId": "A"}],
        **extra,
    }


def _dashboard(*panels: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return seed_dashboard({"uid": "svc", "title": "Svc", "panels": list(panels), **extra})


def _variable(**extra: Any) -> dict[str, Any]:
    return {
        "name": "service",
        "type": "query",
        "datasource": PROMETHEUS,
        "query": "label_values(http_requests_total, job)",
        **extra,
    }


def _annotation(name: str, expr: str = '{job="a"}') -> dict[str, Any]:
    return {"name": name, "enable": True, "datasource": {"type": "loki"}, "expr": expr}


def _before(**extra: Any) -> dict[str, Any]:
    return _dashboard(
        _panel(1, "Request Rate"),
        _panel(2, "Error Rate", fieldConfig={"defaults": {"unit": "short"}}),
        _panel(3, "Latency"),
        templating={"list": [_variable()]},
        annotations={"list": [_annotation("Deployments")]},
        **extra,
    )


def _render(before: dict[str, Any], after: dict[str, Any]) -> str:
    return render_compact_diff("svc", diff_dashboards(before, after))


def _edited(before: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    after = copy.deepcopy(before)
    mutate(after)
    return after


# What the judge reads when something changed.


def test_untouched_dashboard_reads_as_unchanged_and_names_what_was_compared() -> None:
    text = _render(_before(), _before())

    assert 'dashboard UID "svc"' in text
    assert f"{UNCHANGED}. Compared: elements, layout, variables" in text


def test_modified_added_and_removed_panels_are_named_with_their_titles() -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["panels"][0]["targets"][0]["expr"] = "rate(errors[5m])"
        del d["panels"][2]
        d["panels"].append(_panel(4, "Backlog"))

    text = _render(_before(), _edited(_before(), mutate))

    assert (
        'elements: changed=[panel-1 "Request Rate"] added=[panel-4 "Backlog"] '
        'removed=[panel-3 "Latency"]'
    ) in text
    assert 'targets[0].expr "up" -> "rate(errors[5m])"' in text


def test_panels_are_matched_by_identity_not_position() -> None:
    def reorder(d: dict[str, Any]) -> None:
        d["panels"].reverse()

    assert UNCHANGED in _render(_before(), _edited(_before(), reorder))


def test_a_grid_reflow_is_layout_not_an_edit() -> None:
    def push_down(d: dict[str, Any]) -> None:
        d["panels"][1]["gridPos"]["y"] += 8

    text = _render(_before(), _edited(_before(), push_down))

    assert 'layout: gridPos moved on [panel-2 "Error Rate"]' in text
    assert "elements: changed=[] added=[] removed=[]" in text


def test_panels_nested_in_a_row_are_their_own_elements_and_renames_show_both_titles() -> None:
    row = {"id": 10, "type": "row", "title": "Row", "collapsed": True, "panels": [_panel(11, "In")]}

    def rename(d: dict[str, Any]) -> None:
        d["panels"][0]["panels"][0]["title"] = "Renamed"

    text = _render(_dashboard(row), _edited(_dashboard(row), rename))

    assert 'changed=[panel-11 "In" -> "Renamed"]' in text


def test_untitled_panels_are_labelled_as_such() -> None:
    before = _dashboard(_panel(1, ""))

    def retype(d: dict[str, Any]) -> None:
        d["panels"][0]["type"] = "stat"

    assert "changed=[panel-1 (untitled)]" in _render(before, _edited(before, retype))


def test_a_long_multiline_title_stays_on_one_bounded_line() -> None:
    long_title = "x" * 200 + "\nsecond line"
    text = _render(_dashboard(), _dashboard(_panel(1, long_title)))
    short = _render(_dashboard(), _dashboard(_panel(1, "short")))

    lines = text.splitlines()
    assert len(lines) == len(short.splitlines())
    assert all(len(line) < 200 for line in lines)
    assert not any(line.startswith("second line") for line in lines)


def test_variables_are_diffed_by_name() -> None:
    before = _dashboard(_panel(1, "A"), templating={"list": [_variable(), _variable(name="env")]})

    def mutate(d: dict[str, Any]) -> None:
        d["templating"]["list"][0]["multi"] = True
        d["templating"]["list"][1:] = [_variable(name="region")]

    text = _render(before, _edited(before, mutate))

    assert "variables: changed=[service] added=[region] removed=[env]" in text


def test_settings_changes_are_named() -> None:
    before = _dashboard(_panel(1, "A"), description="old")

    def mutate(d: dict[str, Any]) -> None:
        d.update(description="new", tags=["incident"], refresh="30s")

    text = _render(before, _edited(before, mutate))

    assert '  description: "old" -> "new"' in text
    assert 'tags: 1 entr(ies) differ\n    incident added: "incident"' in text
    assert "refresh" in text.split("dashboard settings changed:")[1]


def test_annotations_render_entry_wise_and_ignore_the_builtin_one() -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["annotations"]["list"] = [
            {"builtIn": 1, "name": "Annotations & Alerts", "enable": True},
            _annotation("Deployments", '{job="b"}'),
            _annotation("Rollouts"),
        ]

    text = _render(_before(), _edited(_before(), mutate))

    assert "annotations: 2 entr(ies) differ" in text
    assert 'Deployments changed: {"datasource"' in text and '"expr":"{job=\\"b\\"}"' in text
    assert 'Rollouts added: {"datasource"' in text
    assert "Annotations & Alerts" not in text


@pytest.mark.parametrize(
    ("links", "edit", "seen"),
    [
        pytest.param(
            [{"title": "A", "url": "/a"}, {"title": "B", "url": "/b"}],
            lambda links: links.insert(0, {"title": "New", "url": "/n"}),
            ("links: 1 entr(ies) differ", "New added"),
            id="named entries match by identity",
        ),
        pytest.param(
            [{"url": "/a"}, {"url": "/b"}],
            lambda links: links[1].update(url="/c"),
            ("links: 1 entr(ies) differ", "#1 changed"),
            id="unnamed entries match by position",
        ),
    ],
)
def test_array_settings_are_itemised_per_entry(
    links: list, edit: Callable, seen: tuple[str, str]
) -> None:
    before = _dashboard(links=links)

    text = _render(before, _edited(before, lambda d: edit(d["links"])))

    assert all(line in text for line in seen)


# Saving a dashboard through Grafana rewrites it. None of these are the agent's edits.


def _stamp_and_empty_options(d: dict[str, Any]) -> None:
    d["panels"][0].update(
        pluginVersion="13.2.0-1", fieldConfig={"defaults": {}, "overrides": []}, options={}
    )


def _copy_panel_datasource_to_target(d: dict[str, Any]) -> None:
    # The seed leaves the target's datasource unset; mcp-grafana saves it explicitly.
    d["panels"][0]["targets"][0]["datasource"] = PROMETHEUS


def _load_variable(d: dict[str, Any]) -> None:
    d["templating"]["list"][0].update(
        query={"query": "label_values(http_requests_total,job)", "refId": "StandardVariableQuery"},
        current={"text": "All", "value": "$__all"},
        options=[{"text": "All", "value": "$__all"}],
    )


def _fill_grafana_defaults(d: dict[str, Any]) -> None:
    d.update(graphTooltip=0, liveNow=False, timepicker={}, links=[])


def _add_builtin_annotation(d: dict[str, Any]) -> None:
    d["annotations"]["list"].insert(0, {"builtIn": 1, "name": "Annotations & Alerts"})


def _resolve_link_url(d: dict[str, Any]) -> None:
    d["links"][0]["url"] = "about:blank"


@pytest.mark.parametrize(
    "mutate",
    [
        _stamp_and_empty_options,
        _copy_panel_datasource_to_target,
        _load_variable,
        _fill_grafana_defaults,
        _add_builtin_annotation,
    ],
)
def test_save_artefacts_are_not_edits(mutate: Callable[[dict[str, Any]], None]) -> None:
    assert UNCHANGED in _render(_before(), _edited(_before(), mutate))


def test_a_link_url_resolved_to_about_blank_is_not_an_edit() -> None:
    before = _before(links=[{"title": "Dash", "type": "dashboards"}])

    assert UNCHANGED in _render(before, _edited(before, _resolve_link_url))


def _change_unit(d: dict[str, Any]) -> None:
    d["panels"][1]["fieldConfig"]["defaults"]["unit"] = "s"


def _repoint_target(d: dict[str, Any]) -> None:
    d["panels"][0]["targets"][0]["datasource"] = LOKI


def _repoint_variable(d: dict[str, Any]) -> None:
    d["templating"]["list"][0]["datasource"] = LOKI


def _share_tooltip(d: dict[str, Any]) -> None:
    d["graphTooltip"] = 1


@pytest.mark.parametrize(
    ("mutate", "seen"),
    [
        (_change_unit, 'changed=[panel-2 "Error Rate"] added=[] removed=[]'),
        (_repoint_target, 'changed=[panel-1 "Request Rate"] added=[] removed=[]'),
        (_repoint_variable, "variables: changed=[service] added=[] removed=[]"),
        (_share_tooltip, "dashboard settings changed:\n  graphTooltip:"),
    ],
    ids=["unit", "target-datasource", "variable-datasource", "graph-tooltip"],
)
def test_real_changes_are_seen(mutate: Callable[[dict[str, Any]], None], seen: str) -> None:
    text = _render(_before(), _edited(_before(), mutate))

    assert UNCHANGED not in text
    assert seen in text


# Bounds on what the judge is handed.


def test_field_changes_are_capped_and_say_so() -> None:
    before = _dashboard(_panel(1, "A", **{f"k{i:02d}": 0 for i in range(40)}))

    def bump_all(d: dict[str, Any]) -> None:
        d["panels"][0].update({f"k{i:02d}": 1 for i in range(40)})

    text = _render(before, _edited(before, bump_all))

    assert "capped at" in text and "further fields not compared" in text
    assert "k00" in text and "k39" not in text


def test_a_rewritten_query_is_shown_whole() -> None:
    query = "sum(rate(http_requests_total{job='payment-service',status=~'5..'}[5m])) by (le) + " * 4

    def rewrite(d: dict[str, Any]) -> None:
        d["panels"][0]["targets"][0]["expr"] = query

    assert len(query) > 300
    assert query in _render(_before(), _edited(_before(), rewrite))


def test_a_huge_diff_is_bounded_and_keeps_the_head() -> None:
    before = _dashboard(*[_panel(i, f"Panel {i}") for i in range(1, 200)])

    def rewrite_all(d: dict[str, Any]) -> None:
        for panel in d["panels"]:
            panel["targets"][0]["expr"] = "vector(1)" * 20

    text = _render(before, _edited(before, rewrite_all))

    assert len(text) < 10_000
    assert text.startswith('Structural diff (before vs after) for dashboard UID "svc":')
    assert text.endswith("... (diff truncated)")


# The fact: the dashboard the task seeded against the one saved in Grafana.

SEED = {
    "uid": "svc",
    "title": "Svc",
    "panels": [_panel(1, "Request Rate"), _panel(2, "Error Rate")],
}
CTX = VerifierContext(grafana_url="http://grafana")


def _stored(**overrides: Any) -> dict[str, Any]:
    """The seed as Grafana stores it after the agent rewrote Error Rate's query."""
    saved_panels = copy.deepcopy(SEED["panels"])
    saved_panels[1]["targets"][0]["expr"] = "rate(errors[5m])"
    return {
        "id": 7,
        "uid": "svc",
        "title": "Svc",
        "version": 3,
        "schemaVersion": 41,
        "editable": True,
        "refresh": "",
        "timezone": "browser",
        "tags": [],
        "time": {"from": "now-6h", "to": "now"},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": saved_panels,
        **overrides,
    }


def _problem(fact_selector: dict[str, Any], seeds: list[dict[str, Any]]) -> Problem:
    fact = {"kind": "resource", "resource": "dashboard_diff", **fact_selector}
    return Problem(
        id="demo",
        category="dashboarding",
        statement="x",
        rubric=[{"criterion": "Only Error Rate changed.", "weight": 1, "fact": fact}],
        setup_dashboards=seeds,
    )


def _grade(
    problem: Problem,
    *,
    saved: tuple[dict[str, Any] | None, str],
    cache: dict[str, FactResult] | None = None,
) -> Any:
    with (
        patch("grading.dashboard_snapshot.search_grafana_dashboard_uid", return_value=("svc", "")),
        patch("grading.dashboard_snapshot.fetch_grafana_dashboard_model", return_value=saved),
    ):
        [criterion] = build_judge_criteria(problem, CTX, cache)
    return criterion


@pytest.mark.parametrize("selector", [{"uid": "svc"}, {"title": "Svc"}], ids=["uid", "title"])
def test_diff_fact_compares_the_seed_with_the_saved_dashboard(selector: dict[str, str]) -> None:
    criterion = _grade(_problem(selector, [SEED]), saved=(_stored(), ""))

    assert not criterion.error and not criterion.answer_backed
    assert 'elements: changed=[panel-2 "Error Rate"] added=[] removed=[]' in criterion.prompt_text
    assert "rate(errors[5m])" in criterion.prompt_text


def test_diff_fact_reports_a_dashboard_that_is_missing_from_grafana() -> None:
    criterion = _grade(
        _problem({"title": "Svc"}, [SEED]),
        saved=(None, "Grafana dashboard HTTP 404 for uid=svc."),
    )

    assert criterion.error and criterion.error.startswith("transport:")


def test_diff_fact_without_a_seeded_dashboard_is_a_grading_error() -> None:
    criterion = _grade(_problem({"uid": "svc"}, []), saved=(None, "boom"))

    assert criterion.error and criterion.error.startswith("empty_evidence")


def test_a_graded_diff_can_be_regraded_offline_from_the_saved_truth() -> None:
    problem = _problem({"uid": "svc"}, [SEED])
    cache: dict[str, FactResult] = {}
    online = _grade(problem, saved=(_stored(), ""), cache=cache)

    truth = load_truth(json.loads(json.dumps(dump_truth(cache))))
    [offline] = build_judge_criteria(problem, None, truth, offline=True)

    assert offline.prompt_text == online.prompt_text


@pytest.mark.parametrize(
    "result",
    [
        FactResult(summary="Structural diff", debug={"resource": "dashboard_diff", "n": [1]}),
        FactResult(summary="", debug={}, error="transport: could not load saved dashboard"),
        FactResult(summary="No dashboard was saved.", debug={}, fails=True),
    ],
    ids=["resolved", "error", "fails"],
)
def test_truth_survives_the_save_and_load_a_regrade_goes_through(result: FactResult) -> None:
    saved = json.loads(json.dumps(dump_truth({"fact-key": result})))

    assert load_truth(saved) == {"fact-key": result}
