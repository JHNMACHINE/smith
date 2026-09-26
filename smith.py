"""smith: an agent for training nodes. Takes jobs from a backend's queue and runs them here.

    python smith.py --config smith.toml

The backend never starts a training: it may be a container with no torch and
no GPU, or have no process to start at all. It keeps a queue; smith runs where
the training environment actually is - a laptop's venv, a container, a rented
GPU node - takes a job, starts it, streams its output and says how it went.

**It runs scripts it was given, never commands it was sent.** A job names a
script from the ``[scripts]`` table of this agent's own configuration, and its
parameters become ``--name value`` flags whose names the backend has already
held to what a flag can be. An agent that ran a command line taken from an API
would be a remote shell on every training machine.

**What a job becomes.** A process running the script with

- ``RAVEX_STORAGE_PATH``: a new store under ``runs`` for a fresh run or a
  fork, the run's own store for a resume;
- ``RAVEX_RUN_ID`` and ``RAVEX_NAME`` for a fresh run or a fork, so the run
  the backend shows has the name it was given and an id this agent knows before
  Ravex starts;
- ``RAVEX_FORK_FROM`` and ``RAVEX_FORK_STEP`` for a fork, which is a fresh
  run of its own starting from another run's checkpoint;
- ``RAVEX_METRICS_ENDPOINT``: the backend, so Ravex sends the run there while
  it trains, and ``RAVEX_METRICS_TOKEN`` with the agent's token when
  the backend asks for one.

**Where a run's store lives.** On a disk of this machine under ``runs``, or -
with a ``[storage]`` table - in a bucket, under ``<prefix>/<run id>``, with
the local directory only as Moonclip's staging copy. A rented node
is gone once it stops; a store in the bucket is what lets "Resume" and "Fork"
work on another one. The run reports ``s3://bucket/prefix`` as its address,
and a resume or fork on any agent with the same bucket starts from there. The
bucket's keys come from the environment, ``RAVEX_S3_ACCESS_KEY`` and
``RAVEX_S3_SECRET_KEY``, never from this file; the jobs inherit them.

**The token.** ``SMITH_TOKEN`` in the environment (or ``token`` in the
configuration, for a machine where that is how secrets arrive): sent as a
bearer token on every call, and handed to the runs and to ``ravex ship``.

**On a rented pod** one image serves every machine, so what differs between
them comes from the environment and wins over the file: ``SMITH_BACKEND``,
``SMITH_NAME``, ``SMITH_HARDWARE``, ``SMITH_GPUS``, and ``SMITH_STORAGE_TYPE``,
``_BUCKET``, ``_PREFIX``, ``_ENDPOINT``, ``_REGION``, ``_PATH_STYLE`` for the bucket.

Its output goes to ``runs/_logs/job-<id>.log``, and from there to the backend
a few seconds at a time, numbered line by line, so whoever watches the backend sees it while
it runs - a traceback from a job that died at import included.

**Stopping.** A job cancelled on the backend is noticed at the next report, and
the process is interrupted the way Ctrl-C would: Ravex sees a
``KeyboardInterrupt``, writes its final checkpoint and records the run as
``interrupted``. One that has not exited a minute later is killed.

**What a run could not send.** A run that ends while the backend is down does
not wait for it - Ravex keeps the unsent metrics on disk for the next
execution on that store, and a finished run never gets one. This agent
outlives the run and sits beside its store, so after every job it runs
``ravex ship`` on it, again every half minute until the backend has it all.
Without that, the first end-to-end test left a run showing ``running`` forever
with four fifths of its points missing.

Only the standard library, so it runs in whatever environment the training
does without adding to it. Python 3.11 or newer, for ``tomllib``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Seconds between two polls of the queue, and between two heartbeats.
POLL_SECONDS = 2.0

#: Seconds between two reports on a running job. Also how long a cancel on
#: the backend takes to reach the process, at most.
REPORT_SECONDS = 4.0

#: How long a job that was asked to stop has before it is killed.
STOP_GRACE_SECONDS = 60.0

#: Seconds between two attempts at sending what a finished run left unsent.
SHIP_RETRY_SECONDS = 30.0

#: Most lines of output one post carries, and most bytes read per poll.
LOG_BATCH = 500
LOG_READ = 1 << 20

TIMEOUT = 10.0

#: Runs the script the way ``python script.py`` would, with one difference: a
#: stop request arrives as ``KeyboardInterrupt``. On Windows the only signal
#: one process can send another's group is CTRL_BREAK, whose default is to
#: end the process on the spot - no final checkpoint, and a status left saying
#: "running". Turned into an exception, it unwinds through Ravex like Ctrl-C.
_LAUNCHER = """
import os, runpy, signal, sys

def _stop(*_):
    raise KeyboardInterrupt

for _name in ("SIGBREAK", "SIGTERM"):
    if hasattr(signal, _name):
        signal.signal(getattr(signal, _name), _stop)

_path = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(_path)))
runpy.run_path(_path, run_name="__main__")
"""


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ─── configuration ──────────────────────────────────────────────────


@dataclass
class Config:
    backend: str
    name: str
    hardware: str
    runs: str
    python: str
    scripts: Dict[str, str] = field(default_factory=dict)
    #: Per script, the parameters that change the model itself - its shape,
    #: what its weights mean. A checkpoint cannot continue with a different
    #: one, so the backend refuses to "save" or fork with it changed and
    #: offers a run from scratch instead. Only the script knows which they are.
    model_params: Dict[str, List[str]] = field(default_factory=dict)
    #: The backend's bearer token for agents and runs; None when it asks none.
    token: Optional[str] = None
    #: The bucket runs keep their stores in; None keeps them on this disk.
    storage: Optional["Bucket"] = None
    #: GPUs on this machine. More than one, and a job runs one process per GPU
    #: under torchrun: a node with four GPUs uses all four.
    gpus: int = 1

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        here = os.path.dirname(os.path.abspath(path))

        def resolve(value: str) -> str:
            return os.path.normpath(os.path.join(here, os.path.expanduser(value)))

        scripts: Dict[str, str] = {}
        model_params: Dict[str, List[str]] = {}
        for key, entry in (raw.get("scripts") or {}).items():
            script = entry.get("path") if isinstance(entry, dict) else entry
            if not script:
                raise SystemExit("scripts.%s has no path" % key)
            resolved = resolve(script)
            if not os.path.isfile(resolved):
                raise SystemExit("scripts.%s: %s does not exist" % (key, resolved))
            scripts[key] = resolved
            model_params[key] = [str(name) for name in (entry.get("model") or [])] if isinstance(entry, dict) else []
        if not scripts:
            raise SystemExit("%s names no scripts; there would be nothing to run" % path)
        # The environment wins over the file for what differs between machines
        # running the same image: a rented pod is configured with variables,
        # and its image carries one file for all of them.
        env = os.environ.get
        table = dict(raw.get("storage") or {})
        for key in ("type", "bucket", "prefix", "endpoint", "region", "path_style"):
            if env("SMITH_STORAGE_" + key.upper()):
                table[key] = env("SMITH_STORAGE_" + key.upper())
        storage = Bucket.from_table(table, path)
        return cls(
            backend=str(env("SMITH_BACKEND") or raw.get("backend", "http://127.0.0.1:8600")).rstrip("/"),
            name=str(env("SMITH_NAME") or raw.get("name") or socket.gethostname()),
            hardware=str(env("SMITH_HARDWARE") or raw.get("hardware", "local-cpu")),
            gpus=max(1, int(env("SMITH_GPUS") or raw.get("gpus", 1))),
            runs=resolve(raw.get("runs", "runs")),
            python=raw.get("python") or sys.executable,
            scripts=scripts,
            model_params=model_params,
            token=os.environ.get("SMITH_TOKEN") or raw.get("token") or None,
            storage=storage,
        )


@dataclass
class Bucket:
    """Where the runs' stores go when they do not stay on this machine."""

    type: str
    bucket: str
    prefix: str = ""
    endpoint: Optional[str] = None
    region: Optional[str] = None
    path_style: bool = False

    @classmethod
    def from_table(cls, table: Any, path: str) -> Optional["Bucket"]:
        if not table:
            return None
        kind = str(table.get("type", "r2"))
        if kind not in ("s3", "r2"):
            raise SystemExit("storage.type in %s is %r; a bucket is s3 or r2" % (path, kind))
        if not table.get("bucket"):
            raise SystemExit("storage in %s names no bucket" % path)
        if kind == "r2" and not table.get("endpoint"):
            raise SystemExit("storage in %s is r2 and has no endpoint" % path)
        for name in ("RAVEX_S3_ACCESS_KEY", "RAVEX_S3_SECRET_KEY"):
            # Said now, not by the first job to fail on it.
            if not os.environ.get(name):
                raise SystemExit("storage in %s is a bucket, but %s is not set" % (path, name))
        return cls(
            type=kind,
            bucket=str(table["bucket"]),
            prefix=str(table.get("prefix", "")).strip("/"),
            endpoint=table.get("endpoint"),
            region=table.get("region"),
            # From TOML a bool, from the environment a string: "false" is not true.
            path_style=str(table.get("path_style", False)).lower() in ("true", "1", "yes"),
        )

    def prefix_of(self, run_id: str) -> str:
        return "%s/%s" % (self.prefix, run_id) if self.prefix else run_id

    def prefix_in(self, uri: str) -> Optional[str]:
        """The prefix of ``uri`` if it is a store in this bucket, else None."""
        head = "s3://%s/" % self.bucket
        return uri[len(head):].strip("/") or None if uri.startswith(head) else None

    def environment(self, prefix: str) -> Dict[str, str]:
        """What Ravex needs to keep a store under ``prefix`` in this bucket."""
        env = {
            "RAVEX_STORAGE_TYPE": self.type,
            "RAVEX_STORAGE_BUCKET": self.bucket,
            "RAVEX_STORAGE_PREFIX": prefix,
            "RAVEX_STORAGE_PATH_STYLE": "true" if self.path_style else "false",
        }
        if self.endpoint:
            env["RAVEX_STORAGE_ENDPOINT"] = str(self.endpoint)
        if self.region:
            env["RAVEX_STORAGE_REGION"] = str(self.region)
        return env


# ─── the backend ────────────────────────────────────────────────────


class Unreachable(Exception):
    pass


class Backend:
    def __init__(self, base: str, token: Optional[str] = None) -> None:
        self.base = base
        self._down_since: Optional[float] = None
        self._headers = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = "Bearer " + token

    def call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                answer = json.loads(response.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError("%s %s: %d %s" % (method, path, exc.code, detail)) from None
        except (urllib.error.URLError, OSError) as exc:
            if self._down_since is None:
                self._down_since = time.monotonic()
                log("backend %s unreachable (%s); jobs already running carry on" % (self.base, exc))
            raise Unreachable(str(exc)) from None
        if self._down_since is not None:
            log("backend reachable again")
            self._down_since = None
        return answer


# ─── a job, running ─────────────────────────────────────────────────


def new_run_id() -> str:
    """The same shape Ravex gives a run it names itself."""
    return "r%s-%s" % (time.strftime("%y%m%d%H%M", time.gmtime()), uuid.uuid4().hex[:6])


def flags(params: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key, value in params.items():
        if value is True:
            out.append("--" + key)
        elif value is False or value is None:
            continue
        else:
            out.extend(["--" + key, str(value)])
    return out


class LogTail:
    """A job's output file, followed and sent to the backend.

    Lines are numbered from the start of the file, and the offset only moves
    once the backend has taken a whole read: a batch lost to a timeout is sent
    again with the same numbers, which the backend keeps once. A line still
    being written - no newline yet - waits for the next read, except at the
    end of the job, when whatever is there is the last line.
    """

    def __init__(self, job_id: int, path: str) -> None:
        self.job_id = job_id
        self.path = path
        self.offset = 0
        self.line = 0
        #: The process has exited: nothing more will be written.
        self.done = False

    def send(self, backend: "Backend") -> bool:
        """Send what is new. True once everything written so far has arrived."""
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                data = handle.read(LOG_READ)
        except OSError:
            return self.done
        if not data:
            return True
        full = len(data) == LOG_READ
        end = data.rfind(b"\n")
        complete = data if (self.done and not full) else data[: end + 1]
        if not complete:
            return not full
        lines = complete.decode("utf-8", "replace").splitlines()
        for first in range(0, len(lines), LOG_BATCH):
            backend.call(
                "POST",
                "/api/jobs/%d/logs" % self.job_id,
                {"start": self.line + first, "lines": lines[first : first + LOG_BATCH]},
            )
        self.offset += len(complete)
        self.line += len(lines)
        return not full and len(complete) == len(data)


class Running:
    """A job this agent started, and the process carrying it out."""

    def __init__(self, job: Dict[str, Any], config: Config) -> None:
        self.job = job
        self.id = int(job["id"])
        self.stop_requested_at: Optional[float] = None
        self.last_report = 0.0

        script = config.scripts[job["script"]]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["RAVEX_METRICS_ENDPOINT"] = config.backend
        if config.token:
            env["RAVEX_METRICS_TOKEN"] = config.token
        bucket = config.storage
        source = ""
        if job["mode"] in ("resume", "update", "fork"):
            source = job.get("source_store_uri") or ""
            if source.startswith("s3://"):
                if bucket is None or bucket.prefix_in(source) is None:
                    raise RuntimeError(
                        "the store of run %s is %s, and this agent %s"
                        % (
                            job.get("source_run_id"),
                            source,
                            "keeps no bucket" if bucket is None else "keeps its runs in s3://%s" % bucket.bucket,
                        )
                    )
            elif not os.path.isdir(source):
                raise RuntimeError(
                    "the store of run %s is %r, which is not a directory on this machine"
                    % (job.get("source_run_id"), source)
                )
        if job["mode"] in ("resume", "update"):
            # The same run carrying on, from its newest checkpoint: the store
            # is the run's own, and Ravex resumes from what is in it.
            self.run_id = str(job["source_run_id"])
            if bucket is not None and source.startswith("s3://"):
                # Back in the bucket under the same prefix. The staging copy is
                # this machine's; when it is empty - a different node from the
                # one that trained the run - Ravex brings the store down first.
                store = os.path.join(config.runs, self.run_id)
                env.update(bucket.environment(bucket.prefix_in(source) or self.run_id))
            else:
                store = source
            if job["mode"] == "update":
                # Parameters somebody changed on a run: the script's
                # new ones win over the checkpoint's, or a new learning rate
                # would be read and then replaced by the old one.
                env["RAVEX_KEEP_HYPERPARAMETERS"] = "1"
        elif job["mode"] in ("fresh", "fork"):
            # The id the backend chose when the job was queued, so it could
            # point at the run before the run existed. A backend
            # from before that has none, and the agent makes one up.
            self.run_id = str(job.get("run_ref") or new_run_id())
            store = os.path.join(config.runs, self.run_id)
            if bucket is not None:
                env.update(bucket.environment(bucket.prefix_of(self.run_id)))
            env["RAVEX_RUN_ID"] = self.run_id
            env["RAVEX_NAME"] = str(job["name"])
            if job["mode"] == "fork":
                # A run of its own, in a new store, starting from the parent's
                # checkpoint at that step. Ravex reads the parent's
                # store once - from the bucket, when that is where it is - and
                # pins the step there when it can.
                env["RAVEX_FORK_FROM"] = source
                if job.get("from_step") is not None:
                    env["RAVEX_FORK_STEP"] = str(job["from_step"])
        else:
            raise RuntimeError("this agent does not know how to start a %r job" % job["mode"])
        env["RAVEX_STORAGE_PATH"] = store
        self.store = store
        if job.get("config"):
            # Settings chosen for the run. RAVEX_OVERRIDE and not one
            # RAVEX_<NAME> each: those lose to the script's own decorator,
            # and a changed checkpoint_every would be read and ignored.
            env["RAVEX_OVERRIDE"] = json.dumps(job["config"])

        logs = os.path.join(config.runs, "_logs")
        os.makedirs(logs, exist_ok=True)
        self.log_path = os.path.join(logs, "job-%d.log" % self.id)
        self._log = open(self.log_path, "ab")
        # Not `tail`: that is the method that reads the last lines for a
        # failure's message.
        self.output = LogTail(self.id, self.log_path)
        if config.gpus > 1:
            # torchrun wants a file, not `-c`. One process per GPU, on this
            # machine only; Ravex sees the ranks and checkpoints once for all.
            # A stop reaches every one of them: they share the process group
            # the signal is sent to.
            launcher = os.path.join(logs, "_launcher.py")
            with open(launcher, "w", encoding="utf-8") as handle:
                handle.write(_LAUNCHER)
            command = [
                config.python, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(config.gpus), launcher, script,
            ]
        else:
            command = [config.python, "-c", _LAUNCHER, script]
        command += flags(job.get("params") or {})
        extra: Dict[str, Any] = {}
        if os.name == "nt":
            # Its own group, so CTRL_BREAK reaches it and not this agent.
            extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            extra["start_new_session"] = True
        self.process = subprocess.Popen(
            command,
            env=env,
            cwd=os.path.dirname(script),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **extra,
        )
        log(
            "job %d (%s, %s) started as run %s, pid %d; output in %s"
            % (self.id, job["name"], job["mode"], self.run_id, self.process.pid, self.log_path)
        )

    def stop(self) -> None:
        now = time.monotonic()
        if self.stop_requested_at is None:
            self.stop_requested_at = now
            log("job %d cancelled by the backend; interrupting pid %d" % (self.id, self.process.pid))
            try:
                if os.name == "nt":
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)
            except OSError:
                pass
        elif now - self.stop_requested_at > STOP_GRACE_SECONDS and self.process.poll() is None:
            log("job %d did not stop within %.0fs; killing it" % (self.id, STOP_GRACE_SECONDS))
            self.process.kill()

    def tail(self, lines: int = 6) -> str:
        """The end of the output: on a failure, usually the line that says why."""
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 4096))
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            return ""
        kept = [line for line in text.splitlines() if line.strip()][-lines:]
        return "\n".join(kept)[-1500:]

    def close(self) -> None:
        self._log.close()


# ─── the loop ───────────────────────────────────────────────────────


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend = Backend(config.backend, config.token)
        self.current: Optional[Running] = None
        #: Stores of finished runs whose leftovers still have to reach the
        #: backend, and when to try next.
        self.to_ship: Dict[str, float] = {}
        self.shipping: Optional[tuple] = None
        #: The output of every job still being sent: the running one, and any
        #: that ended while the backend was not answering.
        self.tails: List[LogTail] = []
        #: `ravex ship` reads the token from here, not from its command line,
        #: where any user of the machine can see it.
        self._ship_env = dict(os.environ)
        if config.token:
            self._ship_env["RAVEX_METRICS_TOKEN"] = config.token

    def beat(self) -> None:
        self.backend.call(
            "POST",
            "/api/agents/heartbeat",
            {
                "agent": self.config.name,
                "scripts": sorted(self.config.scripts),
                "model_params": self.config.model_params,
                "hardware": self.config.hardware,
                "host": socket.gethostname(),
                "job_id": self.current.id if self.current else None,
            },
        )

    def report(self, job_id: int, state: str, message: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"state": state}
        if message is not None:
            body["message"] = message
        if run_id is not None:
            body["run_id"] = run_id
        return self.backend.call("PATCH", "/api/jobs/%d" % job_id, body) or {}

    def take(self) -> None:
        job = self.backend.call(
            "POST",
            "/api/jobs/claim",
            {"agent": self.config.name, "scripts": sorted(self.config.scripts), "hardware": self.config.hardware},
        )
        if not job:
            return
        try:
            self.current = Running(job, self.config)
        except Exception as exc:
            log("job %d cannot start: %s" % (job["id"], exc))
            self.report(int(job["id"]), "failed", "the agent could not start it: %s" % exc)
            return
        self.current.last_report = time.monotonic()
        self.tails.append(self.current.output)
        self.report(self.current.id, "running", run_id=self.current.run_id)

    def watch(self) -> None:
        running = self.current
        assert running is not None
        code = running.process.poll()
        if code is not None:
            running.close()
            running.output.done = True
            self.send_logs()
            self.current = None
            if running.stop_requested_at is not None:
                state, message = "failed", "stopped on request"
            elif code == 0:
                state, message = "finished", None
            else:
                state, message = "failed", "exit code %d\n%s" % (code, running.tail())
            log("job %d %s (exit code %d)" % (running.id, "stopped" if running.stop_requested_at else state, code))
            self._report_until_heard(running.id, state, message, running.run_id)
            # Always, not only after an outage: it is how the final status
            # arrives when the run's own last attempt did not, and on a store
            # with nothing left it sends the run document again and no points.
            self.to_ship[running.store] = time.monotonic()
            return
        if running.stop_requested_at is not None:
            running.stop()
            return
        if time.monotonic() - running.last_report >= REPORT_SECONDS:
            running.last_report = time.monotonic()
            answer = self.report(running.id, "running", run_id=running.run_id)
            if answer.get("state") == "cancelled":
                running.stop()

    def send_logs(self) -> None:
        """Send what the jobs printed since the last time; drop a finished
        job's output once all of it has arrived."""
        for tail in list(self.tails):
            try:
                caught_up = tail.send(self.backend)
            except (Unreachable, RuntimeError):
                # Kept where it was: sent again, with the same line numbers,
                # when the backend answers.
                continue
            if caught_up and tail.done:
                self.tails.remove(tail)

    def ship(self) -> None:
        """Send what finished runs left unsent, one store at a time.

        ``ravex ship`` runs as its own process, in the training environment,
        so this agent keeps needing nothing but the standard library - and a
        backend that takes its time does not hold up the queue.
        """
        if self.shipping is not None:
            store, process, first = self.shipping
            code = process.poll()
            if code is None:
                return
            self.shipping = None
            if code == 0:
                del self.to_ship[store]
                if not first:
                    log("what %s had left reached the backend" % store)
            else:
                # Said once per store, not every half minute.
                if first:
                    log(
                        "%s still has metrics the backend has not got; trying again every %.0fs"
                        % (store, SHIP_RETRY_SECONDS)
                    )
                self.to_ship[store] = -(time.monotonic() + SHIP_RETRY_SECONDS)
            return
        now = time.monotonic()
        for store, due in self.to_ship.items():
            # A negative time is a retry: the first failure was already said.
            if abs(due) <= now:
                process = subprocess.Popen(
                    [
                        self.config.python, "-m", "ravex._cli", "ship",
                        "--storage", store, "--endpoint", self.config.backend, "--wait", "60",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    env=self._ship_env,
                )
                self.shipping = (store, process, due >= 0)
                return

    def _report_until_heard(self, job_id: int, state: str, message: Optional[str], run_id: str) -> None:
        # The end of a job is the one report that must not be lost: without it
        # the backend shows a run as going forever.
        while True:
            try:
                self.report(job_id, state, message, run_id)
                return
            except Unreachable:
                time.sleep(POLL_SECONDS * 2)

    def reconcile(self) -> None:
        """Close the jobs the backend still thinks this agent is running.

        A new agent process has no child: whatever it was running died with
        the old one - a crash, a restart of its container. Left alone, those
        jobs would show as running for ever, and a job asked to stop
        would stay "stopping". Said as what it is, with the output that did
        arrive still with the backend.
        """
        try:
            jobs = self.backend.call("GET", "/api/jobs") or []
        except (Unreachable, RuntimeError):
            return
        for job in jobs:
            held = job.get("state") in ("claimed", "running") or (
                job.get("state") == "cancelled" and job.get("finished_at") is None
            )
            if job.get("agent") == self.config.name and held:
                log("job %d was left running by an earlier start of this agent; closing it" % job["id"])
                try:
                    self.report(
                        int(job["id"]),
                        "failed",
                        "the agent restarted while this job was running, and its process went with it",
                    )
                except (Unreachable, RuntimeError):
                    pass

    def run(self) -> None:
        bucket = self.config.storage
        log(
            "agent %s: %d script(s) (%s), %s, backend %s%s, runs in %s"
            % (
                self.config.name,
                len(self.config.scripts),
                ", ".join(sorted(self.config.scripts)),
                self.config.hardware,
                self.config.backend,
                " with a token" if self.config.token else "",
                self.config.runs
                if bucket is None
                else "s3://%s/%s, staged in %s" % (bucket.bucket, bucket.prefix, self.config.runs),
            )
        )
        self.reconcile()
        while True:
            # Before the heartbeat: it needs no answer from the backend to
            # start, and it is exactly the work to do while there is none.
            self.ship()
            self.send_logs()
            try:
                self.beat()
                if self.current is None:
                    self.take()
                else:
                    self.watch()
            except Unreachable:
                pass
            except RuntimeError as exc:
                log(str(exc))
            time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backend's training jobs on this machine.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "smith.toml"),
        help="the agent's configuration (default: smith.toml beside this file)",
    )
    args = parser.parse_args()
    agent = Agent(Config.load(args.config))
    try:
        agent.run()
    except KeyboardInterrupt:
        if agent.current is not None and agent.current.process.poll() is None:
            log("stopping; job %d keeps running as pid %d" % (agent.current.id, agent.current.process.pid))


if __name__ == "__main__":
    main()
