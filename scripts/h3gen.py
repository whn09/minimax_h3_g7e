#!/usr/bin/env python3
"""Flexible MiniMax-H3 client: all three tasks, free geometry, free step count.

Everything the deliverable requires to be a parameter is one:

  --width/--height   exact output geometry (or --short-edge + --aspect)
  --steps            denoise steps
  --duration         clip length in seconds (4..15, the model's own limit)
  --task             t2va | fl2va | ref2va

Examples
--------
  # text -> video+audio, 10 s clip, 16 steps, 864x480
  h3gen.py --width 864 --height 480 --steps 16 --duration 10

  # exact geometry, any 32-aligned pair inside ratio 1:4..4:1 and the area cap
  h3gen.py --width 640 --height 480 --steps 20 --duration 10
  h3gen.py --width 1120 --height 480 --steps 20        # 21:9

  # image -> video+audio (first frame).  --inline sends the bytes in the request body
  h3gen.py --task fl2va --image assets/first.png --inline --width 864 --height 480 --steps 20

  # image + last frame
  h3gen.py --task fl2va --image assets/first.png --last-image assets/last.png --inline

  # reference video (with its soundtrack) -> video+audio.  NOTE: ref2va needs a
  # server started with --model-variant ref2va; it is a different weight partition.
  h3gen.py --task ref2va --ref-video assets/ref.mp4 --inline --port 30030

  # from your laptop against a remote box: --host, and --inline is then required
  h3gen.py --host <box> --task fl2va --image assets/first.png --inline

Condition files reach the server three ways, and the server resolves the `uri` string itself:
`--inline` (bytes in the request body), an http(s):// URL the SERVER fetches, or a plain path that
only works when the server can see it.  Sample materials are in `assets/`.

Geometry note: the two parameter groups are mutually exclusive, by design.
`--short-edge/--aspect` is the released wire contract.  `--width/--height` is not natively a wire
field, so this script picks the most portable form that expresses it (see `geometry()` for the
ladder) and prints which one it used, so a silent substitution is never possible:

  ratio  short_edge + aspect_ratio, one of the 6 released strings -- works on a stock server
  literal  short_edge + aspect_ratio "W:H"      -- fl2va only, no patch needed
  exact  target.width + target.height           -- any task, needs the width/height patch

Force one with --wire ratio|literal|exact when you want to test a specific path.
"""
import argparse, base64, json, math, os, sys, time, urllib.error, urllib.request

CANVAS_MULTIPLE = 32
MAX_PIXELS = 768 * 1344

#: Schemes the server resolves on its own, so they go on the wire untouched. `data:` / `base64://`
#: carry the bytes in the request body; http(s) is fetched BY THE SERVER; a bare path or `file://`
#: is read by the server too, so it only works when client and server share a filesystem.
#: (`minimax_h3_localize_material_uri`, .../model_specific_stages/minimax_h3/material_io.py:761.)
PASSTHROUGH_SCHEMES = ("data:", "base64://", "http://", "https://", "file://",
                       "tar+offset://", "tar+b64header://", "s3://")

#: Only needs to be close enough for the server to pick a temp-file suffix.
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".mp4": "video/mp4", ".mov": "video/quicktime",
               ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}


def material_uri(value, inline):
    """Turn a --image/--ref-video argument into the `uri` string the server will resolve.

    Default is pass-through: a bare path is then read *on the server*. With --inline the file is
    read here and sent as a `data:` URI, which is what a client on a different machine needs.
    """
    if value is None or value.startswith(PASSTHROUGH_SCHEMES):
        return value
    if not inline:
        return value
    path = os.path.expanduser(value)
    if not os.path.isfile(path):
        sys.exit(f"--inline: not a file on this client: {path}")
    media = MEDIA_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    with open(path, "rb") as fh:
        raw = fh.read()
    payload = base64.b64encode(raw).decode("ascii")
    print(f"inlined {os.path.basename(path)}: {len(raw)} B -> data:{media};base64, "
          f"{len(payload)} chars", file=sys.stderr)
    return f"data:{media};base64,{payload}"


#: The only aspect_ratio strings a stock server accepts for t2va/ref2va. The check is a literal
#: string membership test, not a ratio comparison, so "640:480" is refused even though it *is* 4:3.
FINITE_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")


def round32(value):
    return max(CANVAS_MULTIPLE, int(round(float(value) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def canvas_for(short_edge, aspect):
    """Reproduce the server's adapt_shape_v1 math: nominal edge x ratio, area cap, 32px grid."""
    w_r, h_r = (int(x) for x in aspect.split(":"))
    ratio = w_r / h_r
    if ratio >= 1.0:
        nw, nh = short_edge * ratio, float(short_edge)
    else:
        nw, nh = float(short_edge), short_edge / ratio
    if nw * nh > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nw * nh))
        nw, nh = nw * scale, nh * scale
    return round32(nw), round32(nh), ratio

DEFAULT_PROMPT = (
    "At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny "
    "brass instruments, then abruptly file out."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="t2va", choices=["t2va", "fl2va", "ref2va"])
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    g = p.add_argument_group("geometry (either --width/--height, or --short-edge/--aspect)")
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--short-edge", type=int, default=480)
    g.add_argument("--aspect", default="16:9", help="'W:H' or 'auto'")
    g.add_argument("--wire", default="auto", choices=["auto", "ratio", "literal", "exact"],
                   help="how to express --width/--height on the wire (default: most portable)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--duration", type=float, default=5.0, help="clip seconds, 4..15")
    p.add_argument("--seed", type=int, default=1101)
    p.add_argument("--flow-shift", type=float, default=12.0)
    p.add_argument("--audio-flow-shift", type=float, default=3.0)
    # Sending `quality` at all switches H3's cache-dit gate off the generic (env-driven) path:
    # the stage requires "quality" NOT to be an explicit field before it will mount the env
    # config, so `--quality lossless` is the per-request "no cache" control on a cache-enabled
    # server. `high` picks the built-in preset, which is gated to 1344x768 @ 50 steps.
    p.add_argument("--quality", choices=["lossless", "high"],
                   help="omit to leave the server's cache-dit env config in charge")
    # The video request model is pydantic `extra="allow"` and several sampling params are declared
    # request fields, so anything in SamplingParams can be driven per request without restarting
    # the server. The one that matters here is the torch profiler:
    #   --extra profile=true --extra num_profiled_timesteps=2
    # (which also needs SGLANG_DIFFUSION_TORCH_PROFILER_DIR in the server's env).
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="extra top-level request field; JSON-parsed when possible, else a string. "
                        "Repeatable.")
    c = p.add_argument_group("conditions")
    c.add_argument("--image", help="fl2va: first-frame image, or ref2va: subject image")
    c.add_argument("--last-image", help="fl2va: last-frame image")
    c.add_argument("--ref-video", help="ref2va: reference video (its soundtrack is used too)")
    c.add_argument("--ref-audio", help="ref2va: reference audio; duration then comes from it")
    c.add_argument("--inline", action="store_true",
                   help="send the condition files' bytes in the request body as data: URIs. "
                        "Without this, a plain path is resolved ON THE SERVER, which only works "
                        "if it can see that path. http(s):// URIs are fetched by the server and "
                        "pass through either way.")
    p.add_argument("--host", default="127.0.0.1",
                   help="server host; use the box's address to drive it from your laptop "
                        "(then --inline, since the server cannot see your filesystem)")
    p.add_argument("--port", type=int, default=30010)
    p.add_argument("--out", help="output prefix (default derived from the request)")
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def geometry(a):
    """Return (target_dict, predicted_w, predicted_h, wire_form).

    The two parameter groups are mutually exclusive, matching the server contract:
    --width/--height, or --short-edge/--aspect.

    In width/height mode we walk a portability ladder, most-portable first, because "exact canvas"
    is not natively a wire field:

      1. `ratio`    one of the 6 released aspect strings reproduces the canvas exactly -> use it.
                    Works on a completely stock server, so this is always preferred.
      2. `literal`  fl2va only: it is exempt from the finite-ratio allowlist, so an arbitrary
                    "W:H" aspect string is accepted with no patch at all.
      3. `exact`    target.width + target.height. Any task, but needs
                    patches/minimax-h3-target-width-height.patch.

    Step 1 is a canvas comparison, not a ratio comparison, and that distinction matters: 864x480 is
    the 32px rounding of a 16:9 nominal (ratio 1.8, not 1.7778), so reducing the fraction would
    never recover "16:9", while asking "does 16:9 at short edge 480 land on 864x480?" does.
    """
    if (a.width is None) != (a.height is None):
        sys.exit("--width and --height must be given together")
    if a.width is None:
        if a.wire != "auto":
            sys.exit("--wire only applies to --width/--height")
        short_edge, aspect = a.short_edge, a.aspect
        target = {"short_edge": short_edge, "aspect_ratio": aspect}
        if aspect == "auto":
            return target, None, None, "ratio"
        w, h, _ = canvas_for(short_edge, aspect)
    else:
        w, h = a.width, a.height
        for name, v in (("width", w), ("height", h)):
            if v <= 0:
                sys.exit(f"--{name} must be positive")
            if v % CANVAS_MULTIPLE:
                sys.exit(f"--{name}={v} must be a multiple of {CANVAS_MULTIPLE}; every wire form "
                         f"either refuses it or silently rounds it to the 32px grid")
        short_edge = min(w, h)
        canonical = next((c for c in FINITE_ASPECT_RATIOS
                          if canvas_for(short_edge, c)[:2] == (w, h)), None)
        form = a.wire
        if form == "auto":
            form = "ratio" if canonical else ("literal" if a.task == "fl2va" else "exact")
        if form == "ratio":
            if not canonical:
                sys.exit(f"--wire ratio: none of {list(FINITE_ASPECT_RATIOS)} lands on {w}x{h} at "
                         f"short edge {short_edge}; use --wire exact (needs the width/height "
                         f"patch) or --wire literal (fl2va only)")
            target = {"short_edge": short_edge, "aspect_ratio": canonical}
        elif form == "literal":
            if a.task != "fl2va":
                print(f"NOTE: --wire literal is only accepted for fl2va; task {a.task!r} will "
                      f"reject it unless the server has the width/height patch, in which case "
                      f"--wire exact is the right form.", file=sys.stderr)
            target = {"short_edge": short_edge, "aspect_ratio": f"{w}:{h}"}
        else:
            # The patch makes width/height mutually exclusive with short_edge+aspect_ratio and
            # rejects the pair, so this target deliberately carries neither of those keys.
            target = {"width": w, "height": h}

    ratio = w / h
    if not 0.25 <= ratio <= 4.0:
        sys.exit(f"aspect ratio {ratio:.3f} is outside the model's inclusive 1:4..4:1 range")
    if short_edge % CANVAS_MULTIPLE:
        sys.exit(f"short edge {short_edge} must be a multiple of {CANVAS_MULTIPLE}; the server "
                 f"only accepts short edges from SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES (plus 768), "
                 f"and every entry there must be 32-aligned")
    if w * h > MAX_PIXELS:
        print(f"WARNING: {w}x{h} exceeds the {MAX_PIXELS} px cap; a ratio request is scaled down, "
              f"an exact one is refused", file=sys.stderr)
    return target, w, h, ("ratio" if a.width is None else form)


def conditions(a):
    """Build the conditions list, enforcing each task's own contract locally."""
    if a.task == "t2va":
        for flag in ("image", "last_image", "ref_video", "ref_audio"):
            if getattr(a, flag):
                sys.exit(f"t2va takes no conditions; drop --{flag.replace('_', '-')}")
        return []
    if a.task == "fl2va":
        if a.ref_video or a.ref_audio:
            sys.exit("fl2va takes keyframe images only; use --image / --last-image")
        if not (a.image or a.last_image):
            sys.exit("fl2va needs --image (first frame) and/or --last-image (last frame)")
        conds = []
        # Order matters: the server only accepts frame_index signatures [0], [-1] or [0, -1].
        if a.image:
            conds.append({"type": "image", "role": "keyframe",
                          "uri": material_uri(a.image, a.inline), "frame_index": 0})
        if a.last_image:
            conds.append({"type": "image", "role": "keyframe",
                          "uri": material_uri(a.last_image, a.inline), "frame_index": -1})
        return conds
    # ref2va
    if a.last_image:
        sys.exit("ref2va has no keyframes; --last-image does not apply")
    conds = []
    if a.image:
        conds.append({"type": "image", "role": "reference",
                      "uri": material_uri(a.image, a.inline)})
    if a.ref_video:
        conds.append({"type": "video", "role": "reference",
                      "uri": material_uri(a.ref_video, a.inline)})
    if a.ref_audio:
        conds.append({"type": "audio", "role": "reference",
                      "uri": material_uri(a.ref_audio, a.inline)})
    if not conds:
        sys.exit("ref2va needs at least one of --image / --ref-video / --ref-audio")
    return conds


def redacted(payload):
    """Copy of the payload with inlined base64 payloads shortened, for the saved request json."""
    out = dict(payload)
    conds = []
    for cond in out.get("conditions") or []:
        uri = cond.get("uri", "")
        if isinstance(uri, str) and uri.startswith(("data:", "base64://")):
            head = uri[:uri.find(",") + 1] if "," in uri[:4096] else uri[:64]
            cond = {**cond, "uri": f"{head}<{len(uri) - len(head)} base64 chars elided>"}
        conds.append(cond)
    if conds:
        out["conditions"] = conds
    return out


def call(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body[:2000]}")
        sys.exit(1)


def main():
    a = parse_args()
    target, pw, ph, wire = geometry(a)
    conds = conditions(a)
    out = a.out or (f"h3_{a.task}_{pw}x{ph}_{a.steps}st_{int(a.duration)}s"
                    if pw else f"h3_{a.task}_{a.steps}st")

    # ref2va with an audio reference derives duration from the reference probe; sending
    # duration_seconds anyway would override that, so leave it out in exactly that case.
    derive_duration = a.task == "ref2va" and (a.ref_audio or a.ref_video) and not a.image
    if not derive_duration:
        target["duration_seconds"] = a.duration
        if not 4.0 <= a.duration <= 15.0:
            sys.exit("--duration must be within the model's 4..15 s range")

    payload = {
        "model": "MiniMax-H3",
        "prompt": a.prompt,
        "task": a.task,
        "conditions": conds,
        "target": target,
        "num_outputs_per_prompt": 1,
        "num_inference_steps": a.steps,
        "flow_shift": a.flow_shift,
        "audio_flow_shift": a.audio_flow_shift,
        "seed": a.seed,
    }
    if not derive_duration:
        payload["seconds"] = int(a.duration)
    if a.quality:
        payload["quality"] = a.quality
    for kv in a.extra:
        if "=" not in kv:
            sys.exit(f"--extra wants KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            payload[k] = json.loads(v)
        except json.JSONDecodeError:
            payload[k] = v

    with open(out + "_request.json", "w") as f:
        # An inlined condition is megabytes of base64; keep the saved copy readable.
        json.dump(redacted(payload), f, indent=2)
    t0 = time.time()
    resp = call(f"http://{a.host}:{a.port}/v1/videos", payload)
    vid = resp["id"]
    print(f"job {vid}  task={a.task} asked={pw}x{ph} wire={wire} target={json.dumps(target)} "
          f"steps={a.steps}", flush=True)
    while True:
        st = call(f"http://{a.host}:{a.port}/v1/videos/{vid}")
        if st.get("status") == "completed":
            break
        if st.get("status") == "failed":
            with open(out + "_failed.json", "w") as f:
                json.dump(st, f, indent=2)
            print("FAILED:", json.dumps(st)[:2000])
            sys.exit(1)
        time.sleep(0.5)
    wall = time.time() - t0
    with open(out + "_status.json", "w") as f:
        json.dump(st, f, indent=2)

    size = ""
    if not a.no_download:
        urllib.request.urlretrieve(f"http://{a.host}:{a.port}/v1/videos/{vid}/content",
                                   out + ".mp4")
        size = f" mp4={os.path.getsize(out + '.mp4') / 1e6:.2f}MB"
    print(f"RESULT task={a.task} asked={pw}x{ph} wire={wire} steps={a.steps} "
          f"dur={'from-reference' if derive_duration else a.duration} "
          f"wall={wall:.2f}s{size} out={out}.mp4", flush=True)


if __name__ == "__main__":
    main()
