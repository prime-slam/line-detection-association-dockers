"""
The interface of initializing different datasets.
"""
from .synthetic_dataset import SyntheticShapes,synthetic_collate_fn
from .wireframe_dataset import WireframeDataset,wireframe_collate_fn
from .yorkurban_dataset import YorkUrbanDataset,yorkurban_collate_fn
from .images_dataset import ImageCollections, images_collate_fn
# from .holicity_dataset import HolicityDataset
# from .merge_dataset import MergeDataset


def get_dataset(mode="train", dataset_cfg=None):
    """ Initialize different dataset based on a configuration. """
    # Check dataset config is given
    if dataset_cfg is None:
        raise ValueError("[Error] The dataset config is required!")
    
    # Synthetic dataset
    if dataset_cfg["dataset_name"] == "synthetic_shape":
        dataset = SyntheticShapes(
            mode, dataset_cfg
        )
        # Get the collate_fn
        # from sold2.dataset.synthetic_dataset import synthetic_collate_fn
        collate_fn = synthetic_collate_fn

    # Wireframe dataset
    elif dataset_cfg["dataset_name"] == "wireframe":
        dataset = WireframeDataset(
            mode, dataset_cfg
        )

        # Get the collate_fn
        collate_fn = wireframe_collate_fn
    elif dataset_cfg["dataset_name"] == "yorkurban":
        dataset = YorkUrbanDataset(
            mode, dataset_cfg
        )

        # Get the collate_fn
        collate_fn = yorkurban_collate_fn
    # Holicity dataset
    elif dataset_cfg["dataset_name"] == "holicity":
        dataset = HolicityDataset(
            mode, dataset_cfg
        )

        # Get the collate_fn
        from sold2.dataset.holicity_dataset import holicity_collate_fn
        collate_fn = holicity_collate_fn
    
    # Dataset merging several datasets in one
    elif dataset_cfg["dataset_name"] == "merge":
        dataset = MergeDataset(
            mode, dataset_cfg
        )

        # Get the collate_fn
        from sold2.dataset.holicity_dataset import holicity_collate_fn
        collate_fn = holicity_collate_fn
    elif dataset_cfg["dataset_name"] == "general":
        dataset = ImageCollections(mode, dataset_cfg)
        collate_fn = images_collate_fn
    else:
        raise ValueError(
    "[Error] The dataset '%s' is not supported" % dataset_cfg["dataset_name"])

    return dataset, collate_fn
