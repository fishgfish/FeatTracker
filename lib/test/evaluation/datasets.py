from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),
    lasot_extension_subset = DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                         kwargs=dict()),
    otb99_lang=DatasetInfo(module=pt % "otb99lang", class_name="OTB99LangDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_lang=DatasetInfo(module=pt % "lasotlang", class_name="LaSOTLangDataset", kwargs=dict()),

    webuot=DatasetInfo(module=pt % "webuot", class_name="WebUOTDataset", kwargs=dict(split='test')),
    uot100=DatasetInfo(module=pt % "uot100", class_name="UOT100Dataset", kwargs=dict()),
    uot32=DatasetInfo(module=pt % "uot32", class_name="UOT32Dataset", kwargs=dict()),
    hut290=DatasetInfo(module=pt % "hut290", class_name="hut290Dataset", kwargs=dict()),
    utb180=DatasetInfo(module=pt % "utb180", class_name="UTB180Dataset", kwargs=dict()),
    vmat=DatasetInfo(module=pt % "vmat", class_name="VMATDataset", kwargs=dict()),
    uwcot220=DatasetInfo(module=pt % "uwcot220", class_name="UWCOT220Dataset", kwargs=dict()),
    uvot400=DatasetInfo(module=pt % "uvot400", class_name="UVOT400Dataset", kwargs=dict(split='test')),
    watb=DatasetInfo(module=pt % "watb", class_name="WATBDataset", kwargs=dict())
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset