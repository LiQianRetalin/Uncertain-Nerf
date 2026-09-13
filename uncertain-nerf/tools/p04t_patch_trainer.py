"""Create a P04-T-only trainer copy from the pinned server RU example."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED_SHA = "dea00d8a11071789d4afb01c01d5969683b53fc96db20d5582a6c5278bec328c"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"P04T trainer anchor count {source.count(old)} != 1: {old[:70]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: p04t_patch_trainer.py PINNED_SOURCE ISOLATED_TARGET")
    original, target = map(Path, sys.argv[1:])
    raw = original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
        raise RuntimeError("server trainer SHA differs from frozen P04T base")
    if target.exists():
        raise FileExistsError(target)
    source = raw.decode("utf-8")
    source = replace_once(source,
        "        trainloader = torch.utils.data.DataLoader(\n            self.trainset,\n            batch_size=cfg.batch_size,\n            shuffle=True,\n            num_workers=4,\n            persistent_workers=True,\n            pin_memory=True,\n        )\n        trainloader_iter = iter(trainloader)\n",
        "        p04t = None\n        if os.environ.get('P04T_RUN_ID'):\n            from tools.p04t_monitor import P04TMonitor, ReplayableRandomSampler\n            p04t_sampler = ReplayableRandomSampler(self.trainset)\n            trainloader = torch.utils.data.DataLoader(\n                self.trainset, batch_size=cfg.batch_size, sampler=p04t_sampler,\n                num_workers=4, persistent_workers=True, pin_memory=True,\n            )\n        else:\n            trainloader = torch.utils.data.DataLoader(\n                self.trainset, batch_size=cfg.batch_size, shuffle=True,\n                num_workers=4, persistent_workers=True, pin_memory=True,\n            )\n        trainloader_iter = iter(trainloader)\n        if os.environ.get('P04T_RUN_ID'):\n            p04t = P04TMonitor(self, schedulers, p04t_sampler)\n            init_step, trainloader_iter = p04t.start(trainloader, trainloader_iter)\n")
    source = replace_once(source,
        "            loss.backward()\n            if os.environ.get(\"P03_SMOKE_AUDIT\", \"0\") == \"1\" and step == 0:\n",
        "            loss.backward()\n            if p04t is not None:\n                p04t.after_photo_backward(\n                    step, info, ru_state, pixels, camtoworlds, Ks, image_ids,\n                    sh_degree_to_use,\n                )\n            if os.environ.get(\"P03_SMOKE_AUDIT\", \"0\") == \"1\" and step == 0:\n")
    source = replace_once(source,
        "                self.viewer.update(step, num_train_rays_per_step)\n\n    @torch.no_grad()\n    def eval(self, step: int, stage: str = \"val\"):\n",
        "                self.viewer.update(step, num_train_rays_per_step)\n\n            if p04t is not None:\n                p04t.after_normal_update(step)\n                if step == 19999:\n                    print('P04T_CONTROLLED_STOP_20000_UPDATES', flush=True)\n                    break\n\n    @torch.no_grad()\n    def eval(self, step: int, stage: str = \"val\"):\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    print("P04T_PATCHED_TRAINER_SHA256=" + hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
