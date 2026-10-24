"""Profile FLOPs, parameters and inference speed of FeatTracker."""

import argparse
import os
import sys
import time
import importlib

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import torch
from thop import profile
from thop.utils import clever_format


def parse_args():
    parser = argparse.ArgumentParser(description='Profile FeatTracker.')
    parser.add_argument('--config', type=str, default='feattracker', help='YAML config name.')
    parser.add_argument('--template-number', type=int, default=3, help='Number of template frames.')
    parser.add_argument('--warmup', type=int, default=500, help='Warm-up iterations.')
    parser.add_argument('--iters', type=int, default=1000, help='Timed iterations.')
    return parser.parse_args()


def profile_model(model, template, search):
    macs, params = profile(model, inputs=(template, search), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is', macs)
    print('overall params is', params)


def speed_test(model, template, search, warmup, iters):
    print("testing speed ...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(template, search)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = model(template, search)
        torch.cuda.synchronize()
        end = time.time()

    avg_lat = (end - start) / iters
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))
    print("FPS is %.2f fps" % (1. / avg_lat))


def main():
    args = parse_args()

    device = "cuda:0"
    torch.cuda.set_device(device)

    yaml_fname = os.path.join(prj_path, 'experiments/feattracker/%s.yaml' % args.config)
    config_module = importlib.import_module('lib.config.feattracker.config')
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)

    from lib.models import build_feattracker
    model = build_feattracker(cfg, training=False).to(device)

    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    template = torch.randn(1, 3, z_sz, z_sz, device=device)
    search = torch.randn(1, 3, x_sz, x_sz, device=device)
    template_list = [template for _ in range(args.template_number)]
    search_list = [search]

    with torch.no_grad():
        model(template_list, search_list)

    profile_model(model, template_list, search_list)
    speed_test(model, template_list, search_list, args.warmup, args.iters)


if __name__ == '__main__':
    main()
