import os

from lib.test.evaluation.environment import EnvSettings

# Project root: <root>/lib/test/evaluation/local.py -> <root>
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DATA = os.path.join(_ROOT, 'data')
_OUTPUT = os.path.join(_ROOT, 'output')


def local_env_settings():
    settings = EnvSettings()

    settings.prj_dir = _ROOT
    settings.save_dir = _OUTPUT

    settings.results_path = os.path.join(_OUTPUT, 'test/tracking_results')
    settings.segmentation_path = os.path.join(_OUTPUT, 'test/segmentation_results')
    settings.network_path = os.path.join(_OUTPUT, 'test/networks')
    settings.result_plot_path = os.path.join(_OUTPUT, 'test/result_plots')

    # Generic tracking datasets
    settings.otb_path = os.path.join(_DATA, 'otb')
    settings.nfs_path = os.path.join(_DATA, 'nfs')
    settings.uav_path = os.path.join(_DATA, 'uav')
    settings.tc128_path = os.path.join(_DATA, 'TC128')
    settings.tpl_path = ''
    settings.vot_path = os.path.join(_DATA, 'VOT2019')
    settings.vot18_path = os.path.join(_DATA, 'vot2018')
    settings.vot22_path = os.path.join(_DATA, 'vot2022')
    settings.itb_path = os.path.join(_DATA, 'itb')
    settings.tnl2k_path = os.path.join(_DATA, 'tnl2k')
    settings.got10k_path = os.path.join(_DATA, 'got10k')
    settings.got10k_lmdb_path = os.path.join(_DATA, 'got10k_lmdb')
    settings.lasot_path = os.path.join(_DATA, 'lasot')
    settings.lasot_lmdb_path = os.path.join(_DATA, 'lasot_lmdb')
    settings.lasot_extension_subset_path = os.path.join(_DATA, 'lasot_extension_subset')
    settings.trackingnet_path = os.path.join(_DATA, 'trackingnet')
    settings.davis_dir = ''
    settings.youtubevos_dir = ''

    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.tn_packed_results_path = ''

    # Underwater tracking datasets
    settings.webuot_path = os.path.join(_DATA, 'webuot')
    settings.uot100_path = os.path.join(_DATA, 'uot100')
    settings.uot32_path = os.path.join(_DATA, 'uot32')
    settings.hut290_path = os.path.join(_DATA, 'hut290')
    settings.utb180_path = os.path.join(_DATA, 'utb180')
    settings.uvot400_path = os.path.join(_DATA, 'uvot400')
    settings.vmat_path = os.path.join(_DATA, 'vmat')
    settings.uwcot220_path = os.path.join(_DATA, 'uwcot220')
    settings.watb_path = os.path.join(_DATA, 'watb')

    return settings
