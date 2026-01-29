Short version: if C++ inside YOLO decides to terminate(), Python cannot save you. The only reliable fix is to put YOLO in its own process so that when it dies, it doesn’t take the whole Flask app with it.

Let me break it down.

1. Why your current approach will always be fragile

You already did all the “Python-level” good stuff:

try/except

rate limiting

circuit breaker

locks

But the failure is here:

terminate called without an active exception

That is C++ calling std::terminate (or being killed by the runtime). At that point:

No Python exception is raised

No try/except helps

The whole process is going down

As long as YOLO lives in the same process as your Flask app, this kind of crash will always be able to kill the API, no matter what Python patterns you add.

So you need a process boundary, not better exception handling.

2. The core idea: isolate YOLO in its own process

Think of YOLO as a flaky hardware module.

You want:

Flask app = stable controller

YOLO worker = separate thing you can power-cycle without rebooting the PLC

Two main patterns you can use.

Option A: Separate YOLO microservice (cleanest)

Run a second Python script vision_service.py that:

Loads YOLO once at startup

Exposes a tiny HTTP API on localhost:5001, e.g. /detect

Your main Flask app (on :8080 or wherever):

For /api/vision/detect, it calls the YOLO service with requests.post("http://127.0.0.1:5001/detect", ...)

If that HTTP request fails / times out, you return a nice 503 / fallback to the frontend

Use PM2 to manage two processes:

pwa-dobot-plc (Flask + PLC etc.)

vision-service (YOLO only)

Result:

If YOLO’s C++ backend explodes:

Only vision-service dies

PM2 restarts it

Your main app stays alive and just reports “vision unavailable”

This is the industrial-grade pattern.

Option B: Spawn YOLO as a child process (no extra HTTP port)

If you don’t want another HTTP server:

Main app starts a child Python process via multiprocessing.Process or subprocess.Popen that:

Loads YOLO

Listens on a simple IPC channel (queue/pipe/stdin/stdout, ZeroMQ, etc.)

Flask handler sends frames/jobs across that channel, waits for a reply with timeout.

If the child process dies (exit code != 0, no reply in time):

Parent marks vision as down

Parent restarts the worker process in the background

Key point: the YOLO model never runs in the same Python interpreter as Flask. Native crash = child exits, parent survives.

If you go this route with multiprocessing + PyTorch:

Set start method to "spawn" at the top-level, not "fork", to avoid weird torch/fork issues.

3. Make crashes less likely (but don’t rely on this alone)

Even with process isolation, it is worth reducing the chance of YOLO detonating:

Memory / model size

On a Pi, large YOLO models will flirt with OOM.

Use the tiniest model you can (e.g. nano/tiny).

Make sure you’re not leaking tensors each call (no accumulating outputs in a list stored globally).

Single model instance

Load model once at startup, reuse it.

No reloading per request, no creating new models in threads.

No multithreading into YOLO

Ensure only one thread in the worker calls model() at a time.

You already added locks; keep them inside the worker too.

Version pinning

Pick a specific combo of torch + YOLO implementation that behaves on your hardware and freeze it in your requirements.

Randomly mixing versions is asking for C++ asserts.

But: even if you fix all this, there is always the chance of a native crash. That is why the process boundary is non-negotiable if you want robustness.

4. Concrete action plan for your current stack (Pi + Flask + PM2)

If this were my rig, I’d do:

Split the codebase logically

backend/app.py = current Flask + PLC + API (no YOLO)

backend/vision_service.py = Flask (or FastAPI) that only does YOLO

Add endpoints

vision_service.py exposes /detect and /analyze that take the same payloads you already use internally.

app.py replaces the direct YOLO calls with HTTP calls to http://127.0.0.1:5001.

Error path in main app

If requests.post to vision times out or fails:

Log it

Return a JSON: { "status": "vision_unavailable", "reason": "backend_error" }

Frontend can show “Vision temporarily unavailable, please retry”.

PM2 ecosystem file

One entry for pwa-dobot-plc (existing).

New entry for vision-service running vision_service.py.

Let PM2 restart vision-service when it crashes.

Now:

Worst case: YOLO explodes every 5 seconds.

User experience: API stays up, only the vision feature intermittently fails instead of dropping the whole app.