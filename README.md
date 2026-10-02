# smith

An agent for training nodes. It runs where the training is — a laptop's venv, a container,
a rented GPU node — takes jobs from a backend's queue, starts them, streams
their output and says how they went. The backend never starts a training itself.

It runs **scripts it was given, never commands it was sent**: a job names a script
from the `[scripts]` table of smith's own configuration, and its parameters become
`--name value` flags. `smith.py` explains the rest — environment, tokens, where a
run's store lives, stopping, and what happens to metrics a run could not send.

Only the standard library, Python 3.11 or newer.

## Running it

On the host, from the training environment's Python:

```bash
python smith.py --config smith.toml
```

In a container on CPU (`Dockerfile`, config `smith.docker.toml`), or on a
provider's GPU node (`Dockerfile.gpu`, config `smith.gpu.toml`). Both images take
Moonclip and Ravex from one of two places:

```bash
# From PyPI, at MOONCLIP_VERSION / RAVEX_VERSION: this repository alone.
docker build -t smith .

# Compiled from checkouts, for a Ravex that is ahead of PyPI.
docker build -t smith --build-arg LIBS=source \
  --build-context moonclip=../moonclip --build-context ravex=../ravex .
```

## What it expects of a backend

JSON over HTTP, with `Authorization: Bearer <SMITH_TOKEN>` when a token is set:

| call | what for |
|---|---|
| `POST /api/agents/heartbeat` | smith is alive, with its name, hardware and the scripts it can run |
| `POST /api/jobs/claim` | the next queued job this node can run, or nothing |
| `PATCH /api/jobs/{id}` | a job's state: running, finished, failed, and the run id |
| `POST /api/jobs/{id}/logs` | the job's output, numbered line by line |
| `GET /api/jobs` | at start, jobs this node left running before a restart |

A claimed job may carry `secrets`, an object of names and values - an
`HF_TOKEN`, a W&B key. smith puts them in that job's process environment only,
never its own, and never one named `RAVEX_*` or `SMITH_*`; and it replaces
their values with `[secret NAME]` in the output lines and failure messages it
sends back.

A claimed job may also carry `storage`, a bucket with its keys
(`type`, `bucket`, `prefix`, `endpoint`, `region`, `path_style`, `access_key`,
`secret_key`): that job's store goes there instead of the agent's own bucket,
the keys reach only its process, and they are kept out of what is sent back
like secrets.

A claimed job that is one node's part of a run trained on several (Ravex's
outer loop) carries `outer`: `member`, the name of this node's store inside the
run's; `token`, the run's `RAVEX_JOB_TOKEN`, set in the job's environment and
kept out of what is sent back; and, on the node that hosts it, `serve`, a port
on which smith runs `ravex rendezvous` beside the script and stops it after.
The rendezvous address and the run's other settings come in the job's
`config`, as for any run.

The runs are [Ravex](https://github.com/JHNMACHINE/ravex) scripts: smith hands
them their store, name and the backend's address through `RAVEX_*` variables,
and ships what a finished run could not send with `ravex ship`.

## Configuration

On a node, what differs between machines comes from the environment:
`SMITH_BACKEND`, `SMITH_TOKEN`, `SMITH_NAME`,
`SMITH_HARDWARE`, `SMITH_GPUS`, `SMITH_STORAGE_*`, and the
bucket's `RAVEX_S3_ACCESS_KEY` / `RAVEX_S3_SECRET_KEY`.

