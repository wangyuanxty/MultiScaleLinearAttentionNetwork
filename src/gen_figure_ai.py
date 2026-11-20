"""Generate a figure with the GPT-Image endpoint and archive its prompt.

Every call writes two files, so a prompt is never lost:

    temp/<slug>.png            the image
    temp/prompts/<slug>.md     model, size, quality, timestamp, verbatim prompt

The archive is the point.  Prompts are the expensive part to reconstruct, and a
figure without the prompt that produced it cannot be reproduced or revised.

Credentials are read from ~/.baoyu-skills/.env (BOM-tolerant) and can be
overridden by OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_IMAGE_MODEL or by the
matching flags.  The key is never printed.

Usage
-----
    # prompt from a file (frontmatter optional; body is used as the prompt)
    python src/gen_figure_ai.py --prompt-file notes/fig1.md --slug fig1_v3

    # prompt inline
    python src/gen_figure_ai.py --prompt "flat vector ..." --slug test

    # check wiring without spending a generation
    python src/gen_figure_ai.py --prompt-file notes/fig1.md --slug fig1_v3 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request

ENV_PATH = os.path.expanduser("~/.baoyu-skills/.env")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "temp", "ai")
MD_DIR = os.path.join(IMG_DIR, "prompts")
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "high"


def load_env(path: str = ENV_PATH) -> dict[str, str]:
    """Read KEY=VALUE lines, tolerating a UTF-8 BOM and Windows line endings."""
    env: dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with io.open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter, body) for a markdown file that may start with ---."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, parts[2].strip()


def read_prompt(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    if args.prompt_file:
        with io.open(args.prompt_file, encoding="utf-8-sig") as fh:
            meta, body = split_frontmatter(fh.read())
        return body, meta
    if args.prompt:
        return args.prompt.strip(), {}
    raise SystemExit("need --prompt or --prompt-file")


def post(base_url: str, key: str, body: dict, timeout: int):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/images/generations",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def save_image(payload: dict, out_path: str) -> int:
    item = payload["data"][0]
    if "b64_json" in item:
        raw = base64.b64decode(item["b64_json"])
    else:
        with urllib.request.urlopen(item["url"], timeout=120) as response:
            raw = response.read()
    with open(out_path, "wb") as fh:
        fh.write(raw)
    return len(raw)


def write_archive(path: str, meta: dict, prompt: str, body: dict,
                  out_path: str, base_url: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "---",
        f"model: {body['model']}",
        f"size: {body['size']}",
        f"quality: {body['quality']}",
        f"created: {stamp}",
        f"output: {os.path.relpath(out_path, ROOT).replace(os.sep, '/')}",
    ]
    for key, value in meta.items():
        if key not in {"model", "size", "quality", "created", "output"}:
            lines.append(f"{key}: {value}")
    lines += [
        "---",
        "",
        f"Endpoint: `{base_url}`",
        "",
        "## Prompt (verbatim)",
        "",
        prompt,
        "",
        "## Call",
        "",
        "```python",
        f"body = {json.dumps(body, ensure_ascii=False)}",
        "```",
        "",
    ]
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--slug", help="output basename; defaults to the prompt-file stem")
    parser.add_argument("--model")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = load_env()
    key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY", "")
    base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL")
                or env.get("OPENAI_BASE_URL", ""))
    model = (args.model or os.environ.get("OPENAI_IMAGE_MODEL")
             or env.get("OPENAI_IMAGE_MODEL", ""))

    prompt, meta = read_prompt(args)
    slug = args.slug or (os.path.splitext(os.path.basename(args.prompt_file))[0]
                         if args.prompt_file else "figure")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    if not slug.startswith("ai_"):
        slug = "ai_" + slug

    body = {"model": model, "prompt": prompt,
            "size": args.size, "quality": args.quality, "n": 1}

    if not key or not base_url or not model:
        missing = [n for n, v in (("key", key), ("base_url", base_url),
                                  ("model", model)) if not v]
        raise SystemExit("missing: " + ", ".join(missing))

    if args.dry_run:
        print(f"model      {model}")
        print(f"base_url   {base_url}")
        print(f"key        <{len(key)} chars>")
        print(f"size       {args.size}  quality {args.quality}")
        print(f"prompt     {len(prompt)} chars")
        print(f"would write {os.path.join(IMG_DIR, slug + '.png')}")
        print(f"             {os.path.join(MD_DIR, slug + '.md')}")
        return 0

    os.makedirs(MD_DIR, exist_ok=True)
    out_png = os.path.join(IMG_DIR, slug + ".png")
    out_md = os.path.join(MD_DIR, slug + ".md")

    try:
        payload = post(base_url, key, body, args.timeout)
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read()[:400].decode("utf-8", "replace"))
        return 1
    except Exception as exc:                      # noqa: BLE001 - report and stop
        print("FAILED:", type(exc).__name__, str(exc)[:300])
        return 1

    size_bytes = save_image(payload, out_png)
    write_archive(out_md, meta, prompt, body, out_png, base_url)
    print(f"saved {out_png} ({size_bytes} bytes)")
    print(f"archived {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())