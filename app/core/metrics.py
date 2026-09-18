from threading import Lock
from typing import Any

class MetricsStore:
    def __init__(self):
        self._lock = Lock()
        # {metric_name: {label_tuple: count}}
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        # {metric_name: {label_tuple: list_of_values}}
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], list[float]]] = {}
        self._buckets = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]

    def inc_counter(self, name: str, value: float = 1.0, labels: dict[str, Any] | None = None) -> None:
        label_tuple = tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            bucket_map = self._counters.setdefault(name, {})
            bucket_map[label_tuple] = bucket_map.get(label_tuple, 0.0) + value

    def observe_histogram(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        label_tuple = tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            bucket_map = self._histograms.setdefault(name, {})
            bucket_map.setdefault(label_tuple, []).append(float(value))

    def _format_labels(self, label_tuple: tuple[tuple[str, str], ...], extra_label: str | None = None) -> str:
        parts = [f'{k}="{v}"' for k, v in label_tuple]
        if extra_label:
            parts.append(extra_label)
        return "{" + ",".join(parts) + "}" if parts else ""

    def generate_prometheus_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            # Counters
            for name, entries in self._counters.items():
                lines.append(f"# HELP {name} Total count of {name}")
                lines.append(f"# TYPE {name} counter")
                for label_tuple, val in entries.items():
                    lbl_str = self._format_labels(label_tuple)
                    lines.append(f"{name}{lbl_str} {val}")

            # Histograms
            for name, entries in self._histograms.items():
                lines.append(f"# HELP {name} Histogram observations of {name}")
                lines.append(f"# TYPE {name} histogram")
                for label_tuple, values in entries.items():
                    count = len(values)
                    total_sum = sum(values)
                    for b in self._buckets:
                        le_count = sum(1 for v in values if v <= b)
                        lbl_str = self._format_labels(label_tuple, f'le="{b}"')
                        lines.append(f"{name}_bucket{lbl_str} {le_count}")

                    # +Inf bucket
                    lbl_inf = self._format_labels(label_tuple, 'le="+Inf"')
                    lines.append(f"{name}_bucket{lbl_inf} {count}")

                    lbl_base = self._format_labels(label_tuple)
                    lines.append(f"{name}_sum{lbl_base} {round(total_sum, 4)}")
                    lines.append(f"{name}_count{lbl_base} {count}")

        return "\n".join(lines) + "\n"

metrics = MetricsStore()
