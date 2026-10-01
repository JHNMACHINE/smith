# CLAUDE.md

smith: an agent for training nodes. One file, `smith.py`, standard library
only, Python 3.11+. The README says what it does and what it expects of a
backend; this file is how to change it.

## It stands alone

- smith works with **any** backend that speaks the API in the README. It names
  no particular one: no product names, no backend's settings. Its own
  configuration is `SMITH_*` (and the bucket's `RAVEX_S3_*`); a backend that
  starts nodes translates its settings into those.
- **Standard library only.** A dependency would have to be installed in every
  training environment smith runs beside.
- **Scripts it was given, never commands it was sent.** A job names a script
  from `[scripts]` and its parameters become `--name value` flags. Nothing a
  backend sends is ever run as a command line; keep it that way.

## The images

- `Dockerfile` (CPU) and `Dockerfile.gpu` take Moonclip and Ravex from PyPI
  (`LIBS=pypi`, the default) or from checkouts (`LIBS=source` with
  `--build-context moonclip=... ravex=...`).
- **The GPU image is provider-independent**: a slim Python base, never a
  provider's own image, even where that would download faster. The weight
  (torch and its CUDA libraries) is handled inside the image, with layers of
  about a gigabyte.
- **TileLang compiles without a GPU.** nvcc 12.8 comes from NVIDIA's apt
  repository (the pip wheel at 12.x has no nvcc binary; TileLang's extra wants
  CUDA 13, too new for the drivers on rented nodes). The target is a dict,
  `{"kind": "cuda", "arch": "sm_89"}`; Hopper and Blackwell want `sm_90a` /
  `sm_120a`. Compile a kernel in the image before renting a node to try it.
- After pushing an image, check it is really in the registry
  (`docker manifest inspect <tag>`) before anything starts a node with it: a
  missing tag only shows up as a node that never boots.
