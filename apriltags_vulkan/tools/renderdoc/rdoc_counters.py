"""Per-command GPU timings and hardware counters from a RenderDoc capture.

Run inside RenderDoc's UI process, which carries the `renderdoc` module:

  RDOC_CAP=frame.rdc RDOC_OUT=frame.tsv qrenderdoc --python rdoc_counters.py

Writes one row per action: event id, name, GPU duration (us, each action timed on its own, so
transfers that overlap in a real frame are overstated) and, for dispatches, every counter in
COUNTERS that the driver offers (AMD exposes them through GPUPerfAPI). Exits the UI when done.
"""
import os

import renderdoc as rd

COUNTERS = ['ExecutionDuration', 'CSBusyCycles', 'CSWavefrontsLaunched', 'CSLDSBankConflictCycles',
            'L0CacheRequestCount', 'L0CacheHitCount', 'L2CacheRequestCount', 'L2CacheHitCount',
            'FetchSize', 'WriteSize', 'MemUnitBusyCycles', 'MemUnitStalledCycles',
            'WriteUnitStalledCycles', 'GPUBusyCycles']


def main(out):
    rd.InitialiseReplay(rd.GlobalEnvironment(), [])
    cap = rd.OpenCaptureFile()
    cap.OpenFile(os.environ['RDOC_CAP'], '', None)
    _, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
    values = {}
    for r in ctl.FetchCounters([rd.GPUCounter.EventGPUDuration]):
        values.setdefault(r.eventId, {})['us'] = r.value.d * 1e6
    found = {}
    for c in ctl.EnumerateCounters():
        d = ctl.DescribeCounter(c)
        if d.name in COUNTERS and d.name not in found:
            found[d.name] = (c, d)
    for name, (c, d) in found.items():
        for r in ctl.FetchCounters([c]):
            v = r.value.d if d.resultType == rd.CompType.Float else r.value.u64
            values.setdefault(r.eventId, {})[name] = v
    names = [n for n in COUNTERS if n in found]
    sf = ctl.GetStructuredFile()
    out.write('\t'.join(['eid', 'action', 'gpu_us'] + names) + '\n')

    def walk(actions):
        for a in actions:
            v = values.get(a.eventId, {})
            cells = [str(a.eventId), a.GetName(sf), '%.2f' % v.get('us', 0.0)]
            cells += [str(v.get(n, '')) for n in names]
            out.write('\t'.join(cells) + '\n')
            walk(a.children)

    walk(ctl.GetRootActions())
    ctl.Shutdown()
    cap.Shutdown()


with open(os.environ['RDOC_OUT'], 'w') as f:
    try:
        main(f)
    except Exception:
        import traceback
        f.write(traceback.format_exc())
os._exit(0)
