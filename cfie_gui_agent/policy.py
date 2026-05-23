from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class PolicyRule:
    rule_id: str
    text: str
    scope: str = "global"
    source: str = "user"
    severity: str = "normal"
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "text": self.text,
            "scope": self.scope,
            "source": self.source,
            "severity": self.severity,
            "evidence_refs": list(self.evidence_refs),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PolicyUpdate:
    update_id: str
    summary: str
    rules: tuple[PolicyRule, ...]
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "summary": self.summary,
            "reason": self.reason,
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PolicyStore:
    rules: dict[str, PolicyRule] = field(default_factory=dict)
    updates: list[PolicyUpdate] = field(default_factory=list)

    def apply_update(
        self,
        *,
        summary: str,
        constraints: Any,
        reason: str | None = None,
        source: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyUpdate:
        rules = tuple(
            self._rule_from_item(
                item,
                scope=scope,
                source=source,
            )
            for scope, item in _iter_constraint_items(constraints)
        )
        update = PolicyUpdate(
            update_id=f"policy_{uuid4().hex}",
            summary=summary,
            reason=reason,
            rules=rules,
            metadata=dict(metadata or {}),
        )
        for rule in rules:
            self.rules[rule.rule_id] = rule
        self.updates.append(update)
        return update

    def to_context_payload(self) -> dict[str, Any]:
        return {
            "rules": [rule.to_dict() for rule in self.rules.values()],
            "updates": [update.to_dict() for update in self.updates[-10:]],
        }

    def _rule_from_item(
        self,
        item: Any,
        *,
        scope: str,
        source: str,
    ) -> PolicyRule:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("rule") or "").strip()
            if not text:
                text = str(item).strip()
            return PolicyRule(
                rule_id=str(item.get("rule_id") or f"rule_{uuid4().hex}"),
                text=text,
                scope=str(item.get("scope") or scope),
                source=str(item.get("source") or source),
                severity=str(item.get("severity") or "normal"),
                evidence_refs=tuple(str(ref) for ref in item.get("evidence_refs", ()) or ()),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "rule_id",
                        "text",
                        "rule",
                        "scope",
                        "source",
                        "severity",
                        "evidence_refs",
                    }
                },
            )
        return PolicyRule(
            rule_id=f"rule_{uuid4().hex}",
            text=str(item).strip(),
            scope=scope,
            source=source,
        )


def _iter_constraint_items(constraints: Any) -> tuple[tuple[str, Any], ...]:
    if constraints is None:
        return ()
    if isinstance(constraints, dict):
        items: list[tuple[str, Any]] = []
        for scope, value in constraints.items():
            if isinstance(value, list):
                items.extend((str(scope), item) for item in value)
            else:
                items.append((str(scope), value))
        return tuple(items)
    if isinstance(constraints, list):
        return tuple(("global", item) for item in constraints)
    return (("global", constraints),)
