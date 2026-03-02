# GEMM ID Log Diagnostics and Fixes

## Summary

Based on the provided logs, there are several issues to address. This document
captures the root causes, fixes, and verification steps.

## Root Causes

### 1. `gemm_id` is always 0 (most severe)

**Cause:** The code includes the correct `gemm_id` implementation, but the
binary running in your environment was not rebuilt after those changes.

Evidence in code:

- `gemm_id` exists in `MatrixPartition`.
- `gemm_id_count_` is an atomic counter in `ProxySvrImpl`.
- `DispatchMatMulAsync()` assigns `partition->gemm_id = gemm_id_count_`.
- `gemm_id_count_` is incremented after each dispatch.

**Fix:** Rebuild and redeploy the project:

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 2. Log files missing header comments

**Cause:** Log files did not include a header that describes the format, but
`merge_perf_logs.py` expects it.

**Fix implemented:** `InitSeparatePerfLog()` now writes the header comments:

```
# VTIME format: VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
# Throughput format: timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us
```

### 3. `merge_perf_logs.py` syntax error

**Cause:** An extra backtick existed near line 24.

**Fix implemented:**

- Removed the stray character.
- Updated comments to include the `gemm_id` field.
- Improved header parsing.

## Log Formats

### VTIME format

```
VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
```

Fields:

- `VTIME`: event type
- `timestamp_us`: system timestamp (microseconds)
- `device_id`: device ID (0, 1, 2, ...)
- `gemm_id`: global GEMM operation ID (increments per `DispatchMatMulAsync`)
- `phase`: `COMPUTE`, `RECEIVE`, `SEND`
- `event`: `START`, `END`
- `vt_start_us`: virtual time start (microseconds)
- `vt_end_us`: virtual time end (microseconds)
- `vt_duration_us`: virtual time duration (microseconds)

Example:

```
VTIME,1765380077213829,2,0,SEND,END,10098658,10100432,1774
VTIME,1765380077218046,2,0,COMPUTE,START,3024656,3024656,0
VTIME,1765380077218874,2,0,COMPUTE,END,3024656,3025486,830
```

### Throughput format

```
timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_start_us,epoch_end_us,packet_duration_us
```

Fields:

- `timestamp_us`: system timestamp (microseconds)
- `device_id`: device ID
- `gemm_id`: global GEMM operation ID
- `direction`: `UPLOAD`, `DOWNLOAD`
- `bytes`: bytes transferred
- `throughput_b_s`: throughput in bytes per second
- `epoch_start_us`: transfer start time (microseconds)
- `epoch_end_us`: transfer end time (microseconds)
- `packet_duration_us`: packet duration (microseconds)

Example:

```
1765380077219527,2,0,DOWNLOAD,131153,70022.96,1765380077219482,1765380077219482,0
1765380077335074,2,1,DOWNLOAD,131154,131878.83,1765380077219482,1765380077335065,115583
```

## Files Updated (summary)

C++:

- `csrc/backend/proxy_svr.cc`: log `gemm_id` in DEBUG/INFO output.
- `csrc/backend/device_tracker.cc`: add log header comments.
- `morphling/ops/csrc/backend/device_tracker.cc`: keep header in sync.

Python:

- `scripts/merge_perf_logs.py`: fix syntax, update comments, improve header parsing.

## Verification Steps

1. Rebuild and redeploy.
2. Run a GEMM and check the logs:

```bash
head -20 logs/perf_server.log
head -20 logs/perf_device_0.log
```

Expected:

- The first two lines are format comments.
- `gemm_id` increments from 0 in both VTIME and throughput logs.

3. Run the merge script:

```bash
python3 scripts/merge_perf_logs.py logs/ logs/perf_merged.log
```

You should see counts for VTIME and throughput events and the header comments
parsed correctly.

## Next Steps

1. Rebuild.
2. Deploy and run.
3. Verify log headers and `gemm_id` increments.
4. Run `merge_perf_logs.py` to validate merged output.
5. Proceed with multi-device sync analysis.
