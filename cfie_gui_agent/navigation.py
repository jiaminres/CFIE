from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any


class NavigationPlanError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class Point:
    x: int
    y: int

    @classmethod
    def from_value(cls, value: Any) -> "Point":
        if isinstance(value, dict):
            return cls(x=int(value["x"]), y=int(value["y"]))
        x, y = value
        return cls(x=int(x), y=int(y))

    def to_list(self) -> list[int]:
        return [self.x, self.y]


@dataclass(slots=True, frozen=True)
class ObstaclePolygon:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise NavigationPlanError("obstacle must contain at least two points")

    @classmethod
    def from_value(cls, value: Any) -> "ObstaclePolygon":
        return cls(points=tuple(Point.from_value(point) for point in value))

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [point.x for point in self.points]
        ys = [point.y for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def to_list(self) -> list[list[int]]:
        return [point.to_list() for point in self.points]


@dataclass(slots=True, frozen=True)
class NavigationRequest:
    source: Point
    target: Point
    obstacles: tuple[ObstaclePolygon, ...] = ()
    target_label: str | None = None
    arrive_radius: int = 24
    max_seconds: float = 5.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> "NavigationRequest":
        return cls(
            source=Point.from_value(arguments["source"]),
            target=Point.from_value(arguments["target"]),
            obstacles=tuple(
                ObstaclePolygon.from_value(item)
                for item in arguments.get("obstacles", ()) or ()
            ),
            target_label=arguments.get("target_label"),
            arrive_radius=int(arguments.get("arrive_radius", 24)),
            max_seconds=float(arguments.get("max_seconds", 5.0)),
            metadata={
                key: value
                for key, value in arguments.items()
                if key
                not in {
                    "source",
                    "target",
                    "obstacles",
                    "target_label",
                    "arrive_radius",
                    "max_seconds",
                }
            },
        )

    def distance_to_target(self) -> float:
        return hypot(self.source.x - self.target.x, self.source.y - self.target.y)


@dataclass(slots=True, frozen=True)
class NavigationPlan:
    waypoints: tuple[Point, ...]
    request: NavigationRequest
    status: str = "planned"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.request.source.to_list(),
            "target": self.request.target.to_list(),
            "target_label": self.request.target_label,
            "arrive_radius": self.request.arrive_radius,
            "max_seconds": self.request.max_seconds,
            "obstacles": [obstacle.to_list() for obstacle in self.request.obstacles],
            "waypoints": [point.to_list() for point in self.waypoints],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class NavigationPlanner:
    obstacle_margin: int = 24

    def plan(self, request: NavigationRequest) -> NavigationPlan:
        if request.arrive_radius < 1:
            raise NavigationPlanError("arrive_radius must be >= 1")
        if request.max_seconds <= 0:
            raise NavigationPlanError("max_seconds must be positive")
        if request.distance_to_target() <= request.arrive_radius:
            return NavigationPlan(
                waypoints=(request.target,),
                request=request,
                status="already_near_target",
            )

        waypoints: list[Point] = [request.source]
        for obstacle in request.obstacles:
            if _segment_intersects_bounds(
                request.source,
                request.target,
                obstacle.bounds,
                margin=self.obstacle_margin,
            ):
                waypoints.append(_detour_point(obstacle.bounds, self.obstacle_margin))
        waypoints.append(request.target)
        return NavigationPlan(
            waypoints=tuple(_dedupe_points(waypoints)),
            request=request,
            metadata={
                "planner": "bbox_detour",
                "obstacle_margin": self.obstacle_margin,
            },
        )


def _segment_intersects_bounds(
    source: Point,
    target: Point,
    bounds: tuple[int, int, int, int],
    *,
    margin: int,
) -> bool:
    min_x, min_y, max_x, max_y = bounds
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    segment_min_x = min(source.x, target.x)
    segment_max_x = max(source.x, target.x)
    segment_min_y = min(source.y, target.y)
    segment_max_y = max(source.y, target.y)
    return not (
        segment_max_x < min_x
        or segment_min_x > max_x
        or segment_max_y < min_y
        or segment_min_y > max_y
    )


def _detour_point(bounds: tuple[int, int, int, int], margin: int) -> Point:
    min_x, min_y, _max_x, _max_y = bounds
    return Point(x=min_x - margin, y=min_y - margin)


def _dedupe_points(points: list[Point]) -> list[Point]:
    deduped: list[Point] = []
    for point in points:
        if deduped and deduped[-1] == point:
            continue
        deduped.append(point)
    return deduped
