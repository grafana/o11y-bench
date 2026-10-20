"""Deterministic before/after diff of a saved dashboard, rendered compactly for the judge.

A judge handed the whole dashboard cannot reliably tell which panel moved, so the diff is
computed here, keyed by element name (`panel-<id>`). Classic (v1) JSON only. Layout is `gridPos`,
compared apart from the element, so Grafana pushing a panel down to make room is a move, not an
edit.
"""

import json
from dataclasses import dataclass, field
from typing import Any

# Render caps.
MAX_RENDER_CHARS = 8 * 1024
MAX_FIELD_CHANGES = 12
MAX_FIELD_VALUE_LEN = 80
# One entry of an array-valued settings surface, and a saved query. These are the graded object,
# not context for one: at the field cap a rewritten query and its original render as the same
# prefix.
MAX_ENTRY_VALUE_LEN = 600
MAX_ENTRIES = 12
MAX_TITLE_LEN = 60

COMPARED_SURFACES = (
    "elements, layout, variables, title, description, tags, editable, liveNow, graphTooltip, "
    "time, refresh, timezone, timepicker, annotations and dashboard links"
)
SETTINGS_FIELDS = (
    "title",
    "description",
    "tags",
    "editable",
    "liveNow",
    "graphTooltip",
    "time",
    "refresh",
    "timezone",
    "timepicker",
    "annotations",
    "links",
)
# What the task provisioner fills in when a seed leaves a field out (pinned by a test). The
# stored dashboard carries these, so the baseline must too or every seed reads as edited.
SEED_DEFAULTS: dict[str, Any] = {
    "editable": True,
    "refresh": "",
    "timezone": "browser",
    "tags": [],
    "time": {"from": "now-6h", "to": "now"},
    "templating": {"list": []},
    "annotations": {"list": []},
    "panels": [],
}
# Grafana's value for a setting the dashboard leaves unset; a save through the API fills it in.
UNSET_SETTINGS: dict[str, Any] = {"graphTooltip": 0, "liveNow": False}
# Query keys whose text is the graded content of a panel, variable or annotation.
QUERY_KEYS = ("expr", "query", "rawSql", "definition", "expression")
# A missing, non-string and blank title all mean a panel with no header.
UNTITLED = "(untitled)"


@dataclass
class FieldChange:
    path: str
    before: str
    after: str


@dataclass
class EntryChange:
    key: str
    change: str  # added | removed | changed
    before: str = ""
    after: str = ""

    def render(self) -> str:
        match self.change:
            case "added":
                return f"{self.key} added: {self.after}"
            case "removed":
                return f"{self.key} removed: {self.before}"
            case _:
                return f"{self.key} changed: {self.before} -> {self.after}"


@dataclass
class SettingsChange:
    name: str
    before: str
    after: str
    entries: list[EntryChange] = field(default_factory=list)


@dataclass
class DashboardDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    # Elements still present whose gridPos differs.
    moved: list[str] = field(default_factory=list)
    variables_added: list[str] = field(default_factory=list)
    variables_removed: list[str] = field(default_factory=list)
    variables_modified: list[str] = field(default_factory=list)
    settings: list[SettingsChange] = field(default_factory=list)
    field_changes: dict[str, list[FieldChange]] = field(default_factory=dict)
    # Rendering only: element name -> title, before and after the turn.
    titles_before: dict[str, str] = field(default_factory=dict)
    titles_after: dict[str, str] = field(default_factory=dict)

    @property
    def changed_variables(self) -> list[str]:
        return sorted(self.variables_added + self.variables_removed + self.variables_modified)

    @property
    def changed(self) -> bool:
        return bool(
            self.added
            or self.removed
            or self.modified
            or self.moved
            or self.changed_variables
            or self.settings
        )

    def label(self, name: str) -> str:
        """`panel-1 "Error rate"`; a rename renders as both titles; an unknown name stays bare."""
        before, after = self.titles_before.get(name), self.titles_after.get(name)
        if before is not None and after is not None and before != after:
            return f"{name} {quote_title(before)} -> {quote_title(after)}"
        title = after if after is not None else before
        return name if title is None else f"{name} {quote_title(title)}"

    def format_set(self, names: list[str]) -> str:
        # ", " rather than a bare comma because a title can contain a comma.
        return "[" + ", ".join(self.label(name) for name in names) + "]"


def seed_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
    """The dashboard as the task provisioner stores it: the seed over its defaults."""
    return {**SEED_DEFAULTS, **{k: v for k, v in payload.items() if k != "folderId"}}


def diff_dashboards(before: dict[str, Any], after: dict[str, Any]) -> DashboardDiff:
    diff = DashboardDiff()
    before_elements, after_elements = dashboard_elements(before), dashboard_elements(after)
    diff.titles_before = {name: element_title(el) for name, el in before_elements.items()}
    diff.titles_after = {name: element_title(el) for name, el in after_elements.items()}

    diff.removed = sorted(set(before_elements) - set(after_elements))
    diff.added = sorted(set(after_elements) - set(before_elements))
    for name in sorted(set(before_elements) & set(after_elements)):
        before_el, after_el = before_elements[name], after_elements[name]
        if canonical_json(before_el.get("gridPos")) != canonical_json(after_el.get("gridPos")):
            diff.moved.append(name)
        before_body, after_body = normalize_element(before_el), normalize_element(after_el)
        if canonical_json(before_body) != canonical_json(after_body):
            diff.modified.append(name)
            changes: list[FieldChange] = []
            collect_field_changes("", before_body, after_body, changes)
            diff.field_changes[name] = changes

    before_vars, after_vars = dashboard_variables(before), dashboard_variables(after)
    diff.variables_removed = sorted(set(before_vars) - set(after_vars))
    diff.variables_added = sorted(set(after_vars) - set(before_vars))
    diff.variables_modified = sorted(
        name
        for name in set(before_vars) & set(after_vars)
        if canonical_json(normalize_variable(before_vars[name]))
        != canonical_json(normalize_variable(after_vars[name]))
    )
    diff.settings = diff_settings(before, after)
    return diff


def dashboard_elements(dashboard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Panels keyed `panel-<id>`, with panels nested in a row flattened out alongside it."""
    elements: dict[str, dict[str, Any]] = {}

    def visit(panels: Any) -> None:
        for panel in panels if isinstance(panels, list) else []:
            if not isinstance(panel, dict):
                continue
            panel_id = panel.get("id")
            # ponytail: an id-less panel is keyed by position, so inserting one above renumbers it
            name = f"panel-{panel_id}" if panel_id is not None else f"panel-#{len(elements)}"
            elements[name] = {k: v for k, v in panel.items() if k != "panels"}
            visit(panel.get("panels"))

    visit(dashboard.get("panels"))
    return elements


def element_title(element: dict[str, Any]) -> str:
    title = " ".join(str(element.get("title") or "").split())
    return title[:MAX_TITLE_LEN] + "..." if len(title) > MAX_TITLE_LEN else title


def normalize_element(element: dict[str, Any]) -> Any:
    """Drop what no agent authors: layout (compared apart), the plugin build stamp, and empties.

    Grafana fills in empty `fieldConfig`/`options` and the UI stamps `pluginVersion` on save, so
    an untouched panel can differ by them. An empty value reads the same as an absent one.
    """
    body = {k: v for k, v in element.items() if k not in ("gridPos", "pluginVersion")}
    # A target without its own datasource uses the panel's; saving it explicitly is not an edit.
    panel_datasource, targets = body.get("datasource"), body.get("targets")
    if panel_datasource and isinstance(targets, list):
        body["targets"] = [
            {**t, "datasource": t.get("datasource") or panel_datasource}
            if isinstance(t, dict)
            else t
            for t in targets
        ]
    return prune_empty(body)


def prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {k: prune_empty(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if not is_empty(v)}
    if isinstance(value, list):
        return [prune_empty(v) for v in value]
    return value


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def dashboard_variables(dashboard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    templating = dashboard.get("templating")
    items = templating.get("list") if isinstance(templating, dict) else None
    variables: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items if isinstance(items, list) else []):
        if isinstance(item, dict):
            variables[str(item.get("name") or f"#{index}")] = item
    return variables


def normalize_variable(variable: dict[str, Any]) -> Any:
    """Collapse the two ways Grafana writes the same variable query, so loading is not editing.

    A query or datasource variable's resolved `current` and `options` are set by the datasource
    on load. The query comes as a string or as an object around one, and the two disagree on
    whitespace after commas. The datasource stays compared.
    """
    out = dict(variable)
    if out.get("type") in ("query", "datasource"):
        out.pop("current", None)
        out.pop("options", None)
    query = out.get("query")
    if isinstance(query, dict):
        text = next((query[k] for k in ("query", "expr") if isinstance(query.get(k), str)), "")
        out["query"] = canonical_query(text)
    elif isinstance(query, str):
        out["query"] = canonical_query(query)
    return prune_empty(out)


def canonical_query(text: str) -> str:
    return " ".join(text.replace(",", ", ").split())


def diff_settings(before: dict[str, Any], after: dict[str, Any]) -> list[SettingsChange]:
    changes: list[SettingsChange] = []
    for name in SETTINGS_FIELDS:
        before_value, after_value = (normalize_setting(name, d.get(name)) for d in (before, after))
        if canonical_json(before_value) == canonical_json(after_value):
            continue
        changes.append(
            SettingsChange(
                name=name,
                before=truncate_field(before_value),
                after=truncate_field(after_value),
                entries=diff_entries(before_value, after_value),
            )
        )
    return changes


def normalize_setting(name: str, value: Any) -> Any:
    """Scene/loader artefacts of a top-level surface, so a load is not an edit."""
    if isinstance(value, dict) and isinstance(value.get("list"), list):
        value = value["list"]  # annotations: {list: [...]}
    if name == "annotations" and isinstance(value, list):
        # Grafana adds its built-in "Annotations & Alerts" entry on load; no agent authors it.
        value = [e for e in value if not (isinstance(e, dict) and e.get("builtIn"))]
    if name == "links" and isinstance(value, list):
        # An empty link url resolves to about:blank once loaded.
        value = [
            {**e, "url": ""} if isinstance(e, dict) and e.get("url") == "about:blank" else e
            for e in value
        ]
    value = prune_empty(value)
    # Unset, empty and Grafana's default all mean "not set", so a save that fills them is no edit.
    if is_empty(value) or (name in UNSET_SETTINGS and value == UNSET_SETTINGS[name]):
        return None
    return value


def diff_entries(before: Any, after: Any) -> list[EntryChange]:
    """Added, removed and changed entries of an array-valued surface, matched by identity."""
    if not isinstance(before, list) and not isinstance(after, list):
        return []
    before_list = before if isinstance(before, list) else []
    after_list = after if isinstance(after, list) else []
    before_by_key = dict(zip(entry_keys(before_list), before_list, strict=True))
    after_by_key = dict(zip(entry_keys(after_list), after_list, strict=True))
    out = [
        EntryChange(key, "removed", before=truncate_entry(entry))
        for key, entry in before_by_key.items()
        if key not in after_by_key
    ]
    for key, entry in after_by_key.items():
        if key not in before_by_key:
            out.append(EntryChange(key, "added", after=truncate_entry(entry)))
        elif canonical_json(before_by_key[key]) != canonical_json(entry):
            out.append(
                EntryChange(
                    key,
                    "changed",
                    before=truncate_entry(before_by_key[key]),
                    after=truncate_entry(entry),
                )
            )
    return out


def entry_keys(entries: list[Any]) -> list[str]:
    """Name each entry so inserting one at the front does not make every later one modified.

    Falls back to position when entries carry no distinguishing name, or share one.
    """
    names = [entry_name(entry) for entry in entries]
    if all(names) and len(set(names)) == len(names):
        return names
    return [f"#{i}" for i in range(len(entries))]


def entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("name", "title", "uid"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def collect_field_changes(path: str, before: Any, after: Any, out: list[FieldChange]) -> None:
    """Record changed leaf paths. Scalars, and arrays whose length changed, are one entry."""
    if len(out) >= MAX_FIELD_CHANGES or canonical_json(before) == canonical_json(after):
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            key_path = f"{path}.{key}" if path else key
            collect_field_changes(key_path, before.get(key), after.get(key), out)
        return
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        for index, (b, a) in enumerate(zip(before, after, strict=True)):
            collect_field_changes(f"{path}[{index}]", b, a, out)
        return
    truncate = truncate_entry if path.rsplit(".", 1)[-1] in QUERY_KEYS else truncate_field
    out.append(FieldChange(path or "(root)", truncate(before), truncate(after)))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def truncate_field(value: Any) -> str:
    rendered = canonical_json(value)
    if len(rendered) <= MAX_FIELD_VALUE_LEN:
        return rendered
    return rendered[:MAX_FIELD_VALUE_LEN] + "..."


def truncate_entry(value: Any) -> str:
    rendered = canonical_json(value)
    if len(rendered) <= MAX_ENTRY_VALUE_LEN:
        return rendered
    return (
        f"{rendered[:MAX_ENTRY_VALUE_LEN]}... "
        f"({MAX_ENTRY_VALUE_LEN / 1024:.1f}KB of {len(rendered) / 1024:.1f}KB shown)"
    )


def quote_title(title: str) -> str:
    return json.dumps(title, ensure_ascii=False) if title else UNTITLED


def render_compact_diff(uid: str, diff: DashboardDiff, max_len: int = MAX_RENDER_CHARS) -> str:
    """The bounded diff the judge reads in place of the dashboard. At most `max_len` characters."""
    lines = [f'Structural diff (before vs after) for dashboard UID "{uid}":']
    if not diff.changed:
        # Name what was compared; "identical" is a claim about surfaces it never looked at.
        lines.append(f"no changes on any compared surface. Compared: {COMPARED_SURFACES}.")
        return bound_string("\n".join(lines) + "\n", max_len)

    lines.append(f"compared surfaces: {COMPARED_SURFACES}")
    lines.append(
        f"elements: changed={diff.format_set(diff.modified)} "
        f"added={diff.format_set(diff.added)} removed={diff.format_set(diff.removed)}"
    )
    lines.append(
        f"layout: gridPos moved on {diff.format_set(diff.moved)}" if diff.moved else "layout: same"
    )
    lines.append(
        f"variables: changed={diff.format_set(diff.variables_modified)} "
        f"added={diff.format_set(diff.variables_added)} "
        f"removed={diff.format_set(diff.variables_removed)}"
    )
    if diff.settings:
        lines.append("dashboard settings changed:")
        for change in diff.settings:
            if not change.entries:
                lines.append(f"  {change.name}: {change.before} -> {change.after}")
                continue
            lines.append(f"  {change.name}: {len(change.entries)} entr(ies) differ")
            lines.extend(f"    {entry.render()}" for entry in change.entries[:MAX_ENTRIES])
            if len(change.entries) > MAX_ENTRIES:
                lines.append(f"    (and {len(change.entries) - MAX_ENTRIES} more, not itemised)")
    else:
        lines.append(f"dashboard settings changed: none of {', '.join(SETTINGS_FIELDS)}")
    if diff.modified:
        lines.append("field changes:")
        for name in diff.modified:
            changes = diff.field_changes.get(name) or []
            parts = [f"{c.path} {c.before} -> {c.after}" for c in changes]
            if len(changes) >= MAX_FIELD_CHANGES:
                # The walk stops at the cap, so this list is a floor, not the whole set.
                parts.append(f"(capped at {MAX_FIELD_CHANGES}, further fields not compared)")
            lines.append(f"  {diff.label(name)}: {'; '.join(parts) or '(nested change)'}")
    return bound_string("\n".join(lines) + "\n", max_len)


def bound_string(text: str, max_len: int) -> str:
    sentinel = "... (diff truncated)"
    if len(text) <= max_len:
        return text
    return text[: max_len - len(sentinel)] + sentinel
