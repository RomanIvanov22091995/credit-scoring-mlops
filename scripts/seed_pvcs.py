"""Засев PVC данными, моделью и отчётами (подготовка к drift-check/retrain).

Копирует в PVC кластера:
  data/reference/*.csv  -> scoring-data-pvc
  models/*              -> scoring-models-pvc
  reports/drift/*       -> scoring-reports-pvc

Запуск: python scripts/seed_pvcs.py   (требует kubectl + работающий кластер)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

NS = "credit-scoring"

PLAN = [
    ("scoring-data-pvc", [("data/reference", "/app/data/reference")]),
    ("scoring-models-pvc", [("models", "/app/models")]),
    ("scoring-reports-pvc", [("reports/drift", "/app/reports/drift")]),
]


def run(cmd: list[str], **kw) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)
    return res.stdout


def copy_dir_to_pvc(pod: str, local_dir: Path, remote_dir: str, claim: str) -> None:
    """kubectl cp: tar через временный файл."""
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    with tarfile.open(tmp_path, "w:gz") as tar:
        tar.add(local_dir, arcname=remote_dir.strip("/").split("/")[-1])
    parent = str(Path(remote_dir).parent)
    run(["kubectl", "-n", NS, "exec", pod, "--", "mkdir", "-p", remote_dir])
    run(["kubectl", "-n", NS, "cp", str(tmp_path), f"{NS}/{pod}:/tmp/seed.tgz"])
    run(["kubectl", "-n", NS, "exec", pod, "--", "sh", "-c",
         f"cd {parent} && tar xzf /tmp/seed.tgz && rm /tmp/seed.tgz"])
    tmp_path.unlink(missing_ok=True)


def main() -> int:
    # Манифест пизионера: busybox, монтирует все три PVC
    pod_spec = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": "pvc-seeder", "namespace": NS},
        "spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "busy", "image": "busybox:1.36",
                "command": ["sleep", "600"],
                "volumeMounts": [
                    {"name": "data", "mountPath": "/app/data"},
                    {"name": "models", "mountPath": "/app/models"},
                    {"name": "reports", "mountPath": "/app/reports"},
                ],
            }],
            "volumes": [
                {"name": "data", "persistentVolumeClaim": {"claimName": "scoring-data-pvc"}},
                {"name": "models", "persistentVolumeClaim": {"claimName": "scoring-models-pvc"}},
                {"name": "reports", "persistentVolumeClaim": {"claimName": "scoring-reports-pvc"}},
            ],
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(pod_spec, tmp)
        spec_path = tmp.name

    run(["kubectl", "-n", NS, "apply", "-f", spec_path])
    run(["kubectl", "-n", NS, "wait", "--for=condition=Ready", "pod/pvc-seeder", "--timeout=120s"])

    root = Path(".")
    for claim, pairs in PLAN:
        for local_dir, remote_dir in pairs:
            if (root / local_dir).exists():
                copy_dir_to_pvc("pvc-seeder", root / local_dir, remote_dir, claim)
                print(f"OK: {local_dir} -> {claim}:{remote_dir}")

    run(["kubectl", "-n", NS, "exec", "pvc-seeder", "--", "find", "/app", "-type", "f"])
    run(["kubectl", "-n", NS, "delete", "pod", "pvc-seeder"])
    print("PVC засеяны")
    return 0


if __name__ == "__main__":
    sys.exit(main())
