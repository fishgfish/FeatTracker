import os

from lib.config.feattracker.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def _checkpoint_path(save_dir, yaml_name, epoch):
    """Return the checkpoint path, falling back to the legacy ODTrack layout."""
    candidates = [
        os.path.join(save_dir, "checkpoints/train/feattracker/%s/FeatTracker_ep%04d.pth.tar" %
                     (yaml_name, epoch)),
        os.path.join(save_dir, "checkpoints/train/odtrack/%s/ODTrack_ep%04d.pth.tar" %
                     (yaml_name, epoch)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def parameters(yaml_name: str, run_id=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir

    yaml_file = os.path.join(prj_dir, 'experiments/feattracker/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg

    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    epoch = cfg.TEST.EPOCH if run_id is None else run_id
    params.checkpoint = _checkpoint_path(save_dir, yaml_name, epoch)

    params.save_all_boxes = False

    return params
