from .dist_gpipe_pipeline_async import GpipeAsync

# Try to import optional modules
try:
    from .dist_gpipe_pipeline_async_offload import GpipeAsyncOffload
except (ImportError, ModuleNotFoundError):
    GpipeAsyncOffload = None
    print("WARNING: dist_gpipe_pipeline_async_offload module not available")

try:
    from .dist_1f1b_pipeline_async import Pipe1F1BAsync
except (ImportError, ModuleNotFoundError):
    Pipe1F1BAsync = None
    print("WARNING: dist_1f1b_pipeline_async module not available")


def get_pp_module(args, vocab_size, num_classes, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
