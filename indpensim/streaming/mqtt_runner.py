"""MQTT runner — publishes IndPenSim trajectory to a UNS broker.

Plug-in for the learning-sim data pipeline (mosquitto → bridge → redpanda
→ consumer → postgres → grafana). Run side-by-side with learning-sim's
toy ``bioreactor-001`` to compare PT1-lag dynamics vs. real ODE.

Default broker is localhost:1883 (matching learning-sim's docker-compose
mosquitto port). Default equipment id is ``bioreactor-real`` so it shows
up as a separate unit alongside the toy.

Usage:
    python -m indpensim.streaming.mqtt_runner \\
        --broker localhost --port 1883 \\
        --equipment bioreactor-real \\
        --pace fixed:2.0 \\
        --raman-spec 1 --raman-every 15

Pacing default ``fixed:2.0`` matches learning-sim's 2-second scan rate.
At h=0.2h sim sample period, that's an effective acceleration of
2/(0.2*3600) = ~360x — a 230h batch finishes in ~38 minutes wall.

Stops on Ctrl-C. Disconnects cleanly.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import replace

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError as e:                          # pragma: no cover
    raise SystemExit(
        "paho-mqtt not installed. Install with: pip install 'indpensim[mqtt]'"
    ) from e

from indpensim.driver import (
    BatchConfig, CampaignConfig, batch_spec_from_python_rng,
)
from indpensim.io.initial_conditions import load_captured_batch
from indpensim.simulation import simulate_iter
from indpensim.streaming.pacing import paced, parse_pace_spec
from indpensim.streaming.sample import StreamConfig
from indpensim.streaming.uns import (
    UnsConfig, build_batch_start_message, build_messages,
    build_phase_start_message, build_phase_transitions, build_state_message,
)


log = logging.getLogger("indpensim.mqtt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IndPenSim → MQTT (UNS) runner")
    p.add_argument("--broker", default="localhost", help="MQTT broker host")
    p.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    p.add_argument("--site", default="plant1")
    p.add_argument("--area", default="fermentation")
    p.add_argument("--line", default="line1")
    p.add_argument("--equipment", default="bioreactor-real",
                   help="UNS equipment id (default: bioreactor-real)")

    p.add_argument("--pace", default="fixed:2.0",
                   help="pacing: fast | fixed:<sec> | accelerated:<factor> (default fixed:2.0)")
    p.add_argument("--raman-every", type=int, default=15,
                   help="emit Raman spectrum every Nth sample (default 15 = ~30s wall at fixed:2.0)")
    p.add_argument("--sample-period", type=float, default=None,
                   help="simulated seconds between sensor samples (must divide 720; "
                        "default: one per 12-min control step). Pair with "
                        "--pace accelerated:<factor>, e.g. 10 + accelerated:1500")
    p.add_argument("--lab-every", type=int, default=60,
                   help="emit lab measurement every Nth sample (default 60 = 12 sim-h)")

    p.add_argument("--from-capture", action="store_true",
                   help="use captured MATLAB init instead of Python RNG")
    p.add_argument("--seed", type=int, default=42,
                   help="master RNG seed (Python-RNG mode)")
    p.add_argument("--capture-seed", type=int, default=42)
    p.add_argument("--capture-batch", type=int, default=1)

    p.add_argument("--raman-spec", type=int, choices=(0, 1, 2), default=1,
                   help="0=no Raman, 1=record only, 2=close PAA loop")
    p.add_argument("--faults", type=int, choices=range(0, 9), default=0)
    p.add_argument("--variable-length", action="store_true")
    p.add_argument("--qos", type=int, choices=(0, 1, 2), default=0,
                   help="MQTT QoS for process tags (default 0)")
    p.add_argument("--batch-id", type=int, default=1,
                   help="batch_id to advertise on _state and _phase_start")
    p.add_argument("--quiet", action="store_true",
                   help="don't print per-sample progress")
    return p


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_client(broker: str, port: int) -> mqtt.Client:
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    return client


def _publish_messages(client: mqtt.Client, msgs: list, qos: int) -> None:
    for topic, payload in msgs:
        client.publish(topic, payload, qos=qos)


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ---- Build batch spec
    if args.from_capture:
        spec = load_captured_batch(seed=args.capture_seed, batch_index=args.capture_batch)
        log.info("loaded captured batch seed=%d b=%d (T=%dh)",
                 args.capture_seed, args.capture_batch, spec.T)
    else:
        rng = np.random.default_rng(args.seed)
        cfg = CampaignConfig()
        bcfg = BatchConfig(
            faults=args.faults, prbs=0,
            fixed_length=not args.variable_length,
            raman_spec=args.raman_spec,
        )
        spec = batch_spec_from_python_rng(rng, batch_no=args.batch_id, campaign=cfg, batch=bcfg)
        log.info("generated batch from seed=%d (T=%dh, raman_spec=%d)",
                 args.seed, spec.T, args.raman_spec)

    if args.from_capture:
        # Honor user's --raman-spec override on captured batches by patching ctrl_flags
        spec_ctrl = replace(spec.control_flags, Raman_spec=args.raman_spec)
    else:
        spec_ctrl = None  # use spec.control_flags as-is

    uns_cfg = UnsConfig(site=args.site, area=args.area, line=args.line,
                        equipment=args.equipment)
    stream_cfg = StreamConfig(raman_every=args.raman_every, lab_every=args.lab_every,
                              sample_period_s=args.sample_period)
    pacer = parse_pace_spec(args.pace)

    # ---- Connect
    log.info("connecting to MQTT %s:%d", args.broker, args.port)
    client = _make_client(args.broker, args.port)
    log.info("publishing under uns/%s/%s/%s/%s/...", args.site, args.area, args.line, args.equipment)

    # ---- Batch-start event (recipe metadata when attached)
    topic, payload = build_batch_start_message(spec.recipe, uns_cfg,
                                                batch_id=args.batch_id)
    client.publish(topic, payload, qos=1)

    # For legacy (no recipe) runs, keep the old FERMENT phase-start event
    # so downstream consumers that expect one still see it. Recipe runs
    # get real transitions emitted per-sample instead.
    if spec.recipe is None:
        topic, payload = build_phase_start_message(uns_cfg, phase="FERMENT",
                                                    batch_id=args.batch_id,
                                                    phase_index=0)
        client.publish(topic, payload, qos=1)

    # ---- Stream
    stop_requested = False

    def _on_signal(signum, frame):
        nonlocal stop_requested
        log.info("signal %d received — stopping after current sample", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    samples_emitted = 0
    bytes_published = 0
    try:
        iterator = simulate_iter(spec, ctrl_flags=spec_ctrl, stream_config=stream_cfg)
        for sample in paced(iterator, pacer):
            if stop_requested:
                break
            msgs = build_messages(sample, uns_cfg)
            # Always emit a state heartbeat alongside the tags.
            state_topic, state_payload = build_state_message(
                sample, uns_cfg, phase="FERMENT", batch_id=args.batch_id,
            )
            msgs.append((state_topic, state_payload))
            # Recipe-driven runs: fire any transitions that happened at this sample.
            msgs.extend(build_phase_transitions(sample, uns_cfg,
                                                 batch_id=args.batch_id))
            _publish_messages(client, msgs, qos=args.qos)
            samples_emitted += 1
            bytes_published += sum(len(p) for _, p in msgs)
            if not args.quiet and samples_emitted % 20 == 0:
                log.info("k=%d sim_t=%.2fh wall=%.1fs (%d samples, %d msgs total)",
                         sample.k, sample.sim_time_h, sample.wall_time_s or 0.0,
                         samples_emitted, samples_emitted)

        # ---- Phase event: end of batch (legacy path only; recipe runs
        # emit their terminal __COMPLETE__ via build_phase_transitions).
        if spec.recipe is None:
            topic, payload = build_phase_start_message(uns_cfg, phase="HARVEST",
                                                        batch_id=args.batch_id,
                                                        phase_index=1)
            client.publish(topic, payload, qos=1)

    finally:
        log.info("disconnecting (%d samples, %.1f KB published)",
                 samples_emitted, bytes_published / 1024)
        client.loop_stop()
        client.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
