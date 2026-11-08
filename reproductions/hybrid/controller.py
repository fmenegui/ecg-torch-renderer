"""Bounded hybrid worker with one exact selected grid checkpoint."""

import argparse
import hashlib
from pathlib import Path

from reproductions.runpod.controller import Controller
from .recover_checkpoint import CHECKPOINT_SHA256


class HybridController(Controller):
    def provision(self):
        checkpoint = Path("build/grid-recovery/grid-checkpoint.pt")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Recovered checkpoint is missing or changed before provisioning")
        super().provision()
        self.scp(checkpoint, "/workspace/grid-checkpoint.pt")
        actual = self.ssh("sha256sum /workspace/grid-checkpoint.pt").stdout.split()[0]
        if actual != CHECKPOINT_SHA256:
            raise ValueError("On-pod selected checkpoint hash mismatch")
        self.ledger["grid_checkpoint_sha256"] = actual
        self.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plan", help="Explicit separately bounded continuation of the frozen protocol")
    p.add_argument("action", choices=["provision", "execute", "cleanup", "delete-volume"])
    args = p.parse_args()
    controller = HybridController(args.cli, args.out, args.plan or Path(__file__).with_name("plan.json"))
    try:
        if args.action in ["cleanup", "delete-volume"]:
            controller.cleanup(delete_volume=args.action == "delete-volume")
        else:
            getattr(controller, args.action)()
    except BaseException as error:
        controller.ledger["error"] = str(error).replace(controller.key, "[redacted]")
        controller.save()
        raise


if __name__ == "__main__":
    main()
