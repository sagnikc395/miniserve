"""Poll loop used once while the long-context run was live (kept in output/)."""

import json
import time
import urllib.request

peak = {"kv": 0.0, "waiting": 0, "running": 0}
for i in range(48):
    try:
        with urllib.request.urlopen("http://localhost:8000/stats", timeout=2) as r:
            s = json.load(r)
    except Exception:
        time.sleep(5)
        continue
    peak["kv"] = max(peak["kv"], s["kv_usage_ratio"])
    peak["waiting"] = max(peak["waiting"], s["waiting"])
    peak["running"] = max(peak["running"], s["running"])
    line = (
        f"t+{(i + 1) * 5}s running={s['running']} waiting={s['waiting']} "
        f"kv={s['kv_usage_ratio']:.3f} steps={s['steps']}"
    )
    print(line, flush=True)
    if s["kv_usage_ratio"] > 0.24:
        print(f"PEAK NEAR: {peak}", flush=True)
        break
    time.sleep(5)
print("observed peak:", peak, flush=True)
