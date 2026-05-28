# Scale Supoclip WorkerService to Zero When Idle

**Status**: Draft (design)
**Linear**: [ENG-5683](https://linear.app/brand-ninja/issue/ENG-5683/scale-supoclip-workerservice-to-zero-when-idle)
**Authors**: Engineering
**Last updated**: 2026-05-28

---

## 1. Problem

`WorkerService` runs `desiredCount=1` on Fargate 24/7. The worker only does useful
work while a render is in progress; outside of that window — the majority of any
real day — the Fargate task burns ~$50–$150 / month per stage doing nothing.

`BackendService` cannot scale to zero (it serves the HTTP API and webhook
callbacks from BN), but `WorkerService` is purely a consumer of an `arq` Redis
queue. There is no architectural reason it must be hot at all times — only the
*latency* of bringing it back up matters.

This document proposes a backend-managed `1 ↔ 0` scaling pattern that drops
worker compute when there is nothing to do, with a clear contract for how
scale-up and scale-down decisions are made, owned, and bounded.

## 2. Goals & non-goals

**Goals**
- Reduce idle worker compute cost to ~$0 during sustained idle windows.
- Preserve correctness: no SIGTERMs mid-render; no dropped retries.
- Preserve current job throughput when warm (a single worker with `max_jobs=4`).
- Be feature-flagged + rolled out per stage with no big-bang prod cutover.

**Non-goals**
- Scaling `BackendService` (it serves webhooks; would 502 on cold start).
- Scaling `WorkerService` *above* 1 on burst — `max_jobs=4` already absorbs
  the burst ceiling we care about today. Multi-worker target tracking is a
  later follow-up.
- Replacing `arq` with Lambda (15-min cap, no `/tmp` persistence, ML model
  cold-start cost — bad fit for Supoclip's render workload).
- Per-job one-shot Fargate (cold start on *every* job, not just per-session).

## 3. Current state

| Service | Desired | Role | Always-on? |
|---|---|---|---|
| `BackendService` | 1 | FastAPI HTTP + webhook | Yes (must stay) |
| `WorkerService` | 1 | `arq` Redis consumer; renders clips | **No reason it must** |
| `postgres` | 1 | Task DB | Yes |
| `redis` | 1 | `arq` queue | Yes |

- Cluster: `SupoclipCluster` (account `014392677562`, region `ap-southeast-2`).
- Image deploy: `:latest`-tag drift (see `reference_supoclip_prod_latest_drift`).
- Worker config: `max_jobs = 4`, `max_tries = 3`, `job_timeout = 10800`
  (see `backend/src/workers/tasks.py:181-186`).
- ECS task `stopTimeout`: **120 s max** (AWS service limit).

**Topology check required before any scale-to-zero work**: confirm `postgres`
and `redis` are *not* co-located in the `WorkerService` task definition. If
they are, refactor them out before merging this design — scaling `WorkerService`
to zero must not take down the queue or the task DB.

## 4. Critical dependency

**`reference_supoclip_retry_transcript_cache_loss` is amplified by
scale-to-zero.** Today, a SIGTERM mid-render causes `arq` to retry, which
re-downloads the source under a new `/tmp` UUID and skips transcription (the
DB caches `transcript_text`), so `load_cached_transcript_data` misses and
*both* captions and vertical reframe silently fall back. Scale-down would
deliberately SIGTERM a worker; if any render is in flight at that moment,
the same failure mode triggers — and it would now trigger on every busy →
idle transition.

Two ways to defuse this, both acceptable:

1. **Block scale-down while `in_flight > 0`** (recommended; cheap; baked
   into this design). The worker is only SIGTERM'd when it has nothing to
   lose. Strictly necessary.
2. **Land ENG-5675** (Supoclip consumes BN's ElevenLabs transcript via
   source ref id). Removes the on-disk word-cache as a thing that *can* be
   lost on retry. Best long-term fix; bigger work; not required for this
   ticket but worth sequencing alongside.

Mitigation (1) is sufficient on its own. Mitigation (2) is a complementary
follow-up.

## 5. Approach: backend-as-scaler

`BackendService` owns *all* scaling decisions. Single authority means simple
reasoning, one place to read for "why is the worker up/down right now", no
two-writer race conditions.

### 5.1 Scale UP — on demand

```
POST /tasks
  → create task (existing)
  → enqueue job in Redis (existing)
  → asyncio.create_task(ensure_worker_running())   ← new, fire-and-forget
  → return 200
```

`ensure_worker_running()`:

```python
async def ensure_worker_running() -> None:
    """Idempotent: bumps WorkerService desired to 1 iff currently 0."""
    svc = await ecs.describe_services(
        cluster=CLUSTER,
        services=[WORKER_SERVICE_NAME],
    )
    desired = svc["services"][0]["desiredCount"]
    if desired >= 1:
        return  # already up or coming up — nothing to do
    await ecs.update_service(
        cluster=CLUSTER,
        service=WORKER_SERVICE_NAME,
        desiredCount=1,
    )
    logger.info("Worker scaled up from 0 → 1 for incoming job")
```

- Fire-and-forget so `POST /tasks` doesn't pay the ECS API round-trip.
- Idempotent: 10 simultaneous tasks → 10 simultaneous `UpdateService` calls
  → ECS dedupes → worker boots once.
- Failure mode: ECS API call fails → log and continue. The job is queued in
  Redis; the periodic loop (§ 5.2) will scale up on the next pass.
- Never fails the request — eventual consistency.

### 5.2 Scale DOWN — periodic idle-detection loop

Background task in `BackendService.lifespan`, ticking every 60 s:

```python
async def idle_scaler_loop() -> None:
    last_busy_at = time.time()
    while True:
        await asyncio.sleep(60)
        if not SCALE_TO_ZERO_ENABLED:
            continue
        queue_depth = await arq_queue_depth(redis)
        in_flight = await arq_in_flight_count(redis)

        # Retry safety net: queued work + worker absent → bring worker back.
        if queue_depth > 0 and await worker_desired() == 0:
            await ensure_worker_running()
            last_busy_at = time.time()
            continue

        if queue_depth > 0 or in_flight > 0:
            last_busy_at = time.time()  # reset idle clock
            continue

        idle_for = time.time() - last_busy_at
        if idle_for >= IDLE_TIMEOUT_SECONDS and await worker_desired() >= 1:
            await scale_worker_to_zero()
            logger.info("Worker scaled 1 → 0 after %ds idle", idle_for)
```

Key properties:

- **Single decision-maker.** The worker never scales itself. Avoids the
  race "worker decides to die → job arrives → backend doesn't know worker
  is about to be SIGTERM'd".
- **`in_flight` is the safety interlock.** As long as a render is running,
  the idle clock stays reset; we never SIGTERM a busy worker.
- **Retries are handled.** If an `arq` retry lands in Redis after a worker
  crash and `desired=0`, the next loop tick scales the worker back up.

### 5.3 Tracking `in_flight`

`arq` doesn't expose in-flight count over Redis natively; we add a tiny
heartbeat using `arq`'s `on_job_start` / `on_job_end` hooks:

```python
# backend/src/workers/tasks.py

async def on_job_start(ctx):
    await ctx["redis"].incr(IN_FLIGHT_KEY)

async def on_job_end(ctx):
    await ctx["redis"].decr(IN_FLIGHT_KEY)

class WorkerSettings:
    on_job_start = on_job_start
    on_job_end = on_job_end
    # ... existing settings
```

`IN_FLIGHT_KEY = "supoclip:worker:in_flight"`. Backend reads it via `GET`.

**Crash safety**: if the worker is SIGKILL'd mid-job, the counter never
decrements. Cap with a short TTL on the key (e.g. `EXPIRE 600` refreshed by
the worker's existing heartbeat ping), or reset to 0 on `on_startup`. Pick
one in implementation.

### 5.4 Reading queue depth

```python
async def arq_queue_depth(redis) -> int:
    # arq stores pending jobs in the default queue key.
    return await redis.zcard("arq:queue")
```

Confirm the actual key name at implementation time (depends on arq version).

## 6. Cold-start UX

First job after an idle window pays:

| Step | Duration |
|---|---|
| Fargate task launch (image pull from ECR cache) | ~20–40 s |
| `arq` worker boot + Python startup + Redis connect | ~5–15 s |
| First ML model load (MediaPipe / DNN face detectors) | ~10–30 s |
| **Total** | **~30–90 s** |

In BN's create flow this lands as ~30–90 s of additional "Hang tight…" before
the first render starts. Subsequent jobs in the same warm window are unaffected.

Acceptable for current usage patterns (low-traffic, mostly demo/POC). If we
ship to a higher-volume customer later, raise `IDLE_TIMEOUT_SECONDS`
aggressively or bump `desiredCount` floor to 1 in that environment.

Optional follow-up (not in this ticket): surface a `task.worker_state` field
(`'cold_starting' | 'warm'`) so the BN frontend can display "Warming up
clipper…" copy when cold.

## 7. Files & components

| # | Change | File / target |
|---|---|---|
| 1 | ECS client + `ensure_worker_running()` + `scale_worker_to_zero()` | `backend/src/services/ecs_scaler.py` (new) |
| 2 | Hook into `POST /tasks` (fire-and-forget scale-up) | `backend/src/api/routes/tasks.py` |
| 3 | Idle-detection background loop | `backend/src/main.py` (FastAPI lifespan) |
| 4 | In-flight heartbeat via `on_job_start` / `on_job_end` | `backend/src/workers/tasks.py` |
| 5 | IAM: BackendService task role + `ecs:DescribeServices`, `ecs:UpdateService` on `service/SupoclipCluster/WorkerService` | infra (verify stack: SST / CDK / Terraform) |
| 6 | Env config: `WORKER_SCALE_TO_ZERO_ENABLED`, `WORKER_IDLE_TIMEOUT_SECONDS`, `WORKER_SERVICE_NAME`, `SUPOCLIP_CLUSTER_NAME` | env vars + `config.py` |
| 7 | Verify `postgres` / `redis` not co-located in `WorkerService` task def | infra inspection (pre-merge gate) |

**Effort**: ~4 days focused + ≥ 1 week preprod soak before prod enablement.

## 8. Edge cases

| Case | Behaviour |
|---|---|
| 10 jobs arrive in 1 s | All `POST /tasks` fire `ensure_worker_running`; ECS dedupes the `UpdateService` calls; worker boots once. |
| Backend can't reach ECS API | Log and continue; `POST /tasks` returns 200 (job is queued). Periodic loop retries the scale-up on the next tick. |
| Worker crashes mid-job (OOM, segfault) | `arq` retry pushes job back to Redis; in-flight counter never decrements (TTL or `on_startup` reset cleans up); periodic loop sees `queue_depth > 0`, scales worker up. |
| Scale-down decision races a new arrival | New arrival's `POST /tasks` calls `UpdateService(desired=1)` atomically; ECS accepts. If SIGTERM has already been issued, the in-flight precondition was 0 (loss-free); the new arrival waits for a fresh task to boot. |
| ECS `stopTimeout` 120s vs render time 3–5 min | Never scale down while `in_flight > 0`. SIGTERM only ever lands on a worker doing nothing. |
| Postgres / Redis co-located in WorkerService task def | **Gate**: refactor before merge. WorkerService scale-to-zero cannot take the queue / DB down with it. |
| Cold start UX | Accepted; users see ~30–90s additional "Hang tight…" on the first post-idle job. |

## 9. Rollout

1. **Phase 0 — Gate**: verify postgres/redis topology; confirm in-flight
   interlock in code review.
2. **Phase 1 — Preprod, flag off**: ship code with
   `WORKER_SCALE_TO_ZERO_ENABLED=false`. Manual end-to-end test by flipping
   the flag locally and observing behaviour.
3. **Phase 2 — Preprod, flag on**: flip flag on in preprod for a week.
   Watch CloudWatch task lifecycle events + arq metrics.
4. **Phase 3 — Prod opt-in**: enable in prod during off-hours initially.
   Tune `IDLE_TIMEOUT_SECONDS` based on observed traffic.
5. **Phase 4 — Prod 24/7**: full enablement.

## 10. Observability

CloudWatch dashboard tracking:
- `WorkerService` `desiredCount`, `runningCount` over time.
- `arq:queue` depth (publish from backend every 60 s).
- In-flight counter.
- Cold-start duration (worker emits a CloudWatch metric on `on_startup`
  with `time_since_first_log_line` or similar).
- Cost per stage (Fargate task-hours).

Alarms:
- `queue_depth > 0 AND desiredCount == 0` for > 5 min → page (scale-up
  broken).
- Cold-start duration P99 > 120 s → warn (image bloat / ECR pull slow).

## 11. Open questions

1. **Infra-as-code stack** — confirm SST / CDK / Terraform before scoping
   the IAM PR shape.
2. **Postgres / Redis topology** — confirm they live in separate ECS
   services, not in `WorkerService`'s task def. Gate.
3. **Final `IDLE_TIMEOUT_SECONDS` default** — proposing 600 (10 min).
   Conservative enough to avoid thrashing under bursty patterns; aggressive
   enough to actually save money.
4. **`in_flight` counter TTL** — bare counter vs counter-with-TTL vs
   per-job-ID set with TTL. Pick the simplest that survives a SIGKILL.
5. **Single-worker `max_jobs=4` ceiling** — does the design need to allow
   scaling to *N* workers for burst, or is `max_jobs=4` per worker
   sufficient for our anticipated ceiling? (If yes: out of scope for this
   ticket. If no: add target-tracking on queue depth as a Phase 5.)

## 12. Out of scope (potential follow-ups)

- **Scale > 1 worker on burst** (target-tracking on queue depth). Not
  needed at current concurrency.
- **`BackendService` scale-to-zero** — would 502 BN webhooks. Don't.
- **Lambda workers** — 15-min cap, `/tmp` non-persistence, ML model
  cold-start cost. Don't.
- **One-shot Fargate task per render** — cold-start *per job* instead of
  per session. Only worth it if jobs are very rare.
- **Cross-stage scaling** (single worker fleet serving preprod + prod).
  Premature; revisit if costs warrant it.

## 13. Rollback

If scale-to-zero behaves badly in any phase:
- Set `WORKER_SCALE_TO_ZERO_ENABLED=false` in the environment.
- Backend stops calling `UpdateService`; current worker stays at whatever
  desired count is set.
- Manually `UpdateService --desired-count 1` to restore the always-on state.
- No data migration, no schema change to revert.

The feature is fully reversible at the env-var level.
