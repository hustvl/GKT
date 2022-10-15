from . import nuscenes_dataset
from . import nuscenes_dataset_generated
from . import nuscenes_dataset_generated_setting1


MODULES = {
    'nuscenes': nuscenes_dataset,
    'nuscenes_generated': nuscenes_dataset_generated,
    'nuscenes_generated_setting1': nuscenes_dataset_generated_setting1
}


def get_dataset_module_by_name(name):
    return MODULES[name]
