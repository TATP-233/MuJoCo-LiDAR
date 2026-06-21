import json
import sys
from pathlib import Path

from benchmark_core import _TRACE_SCENES, benchmark_trace_rays

BASELINE_FILE = Path(__file__).parent / "baselines" / "baseline.json"
REGRESSION_THRESHOLD = 0.05  # 5% 性能下降阈值


def load_baseline():
    if not BASELINE_FILE.exists():
        return None
    with open(BASELINE_FILE) as f:
        return json.load(f)


def save_baseline(results):
    BASELINE_FILE.parent.mkdir(exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(results, f, indent=2)


def check_regression():
    baseline = load_baseline()

    # 运行当前基准测试
    current = {
        f"{scene_name}/cpu": benchmark_trace_rays(scene_name, scene_file, backend="cpu")
        for scene_name, scene_file in _TRACE_SCENES.items()
    }

    if baseline is None:
        print("No baseline found, saving current results as baseline")
        save_baseline(current)
        return 0

    # 检查回归
    failed = False
    for backend, result in current.items():
        if result is None:
            continue

        baseline_result = baseline.get(backend)
        if baseline_result is None:
            continue

        current_time = result["mean_ms"]
        baseline_time = baseline_result["mean_ms"]
        regression = (current_time - baseline_time) / baseline_time

        print(
            f"{backend}: {current_time:.2f}ms (baseline: {baseline_time:.2f}ms, {regression * 100:+.1f}%)"
        )

        if regression > REGRESSION_THRESHOLD:
            print(f"  ❌ REGRESSION DETECTED: {regression * 100:.1f}% slower")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(check_regression())
