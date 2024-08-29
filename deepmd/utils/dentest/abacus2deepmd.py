import dpdata
import glob
import numpy as np
import os
import shutil
from typing import List, Dict
from dpdata.data_type import (
    Axis,
    DataType,
)
from comp import dump
dpdata.deepmd.comp.dump = dump


def process_density_dataset(frame_paths: List[str], output_path: str, portable=True):
    """
    Process DFT density calculation dataset for multiple systems.
    
    :param frame_path: all frames' paths in a system (e.g. ["/path/to/Mg0Al13Cu3/0", "/path/to/Mg0Al13Cu3/1", ...])
    :param output_path: Path to save the processed dataset
    :param portable: bool. if True, create datasetsin output_path.
    """
    density_data_type = DataType(
        "density",
        np.ndarray,
        shape=(Axis.NFRAMES, 1),
        required=False,
    )
    dpdata.System.register_data_type(density_data_type)
    dpdata.LabeledSystem.register_data_type(density_data_type)
    ms = dpdata.MultiSystems()

    for frame in frame_paths:
        s = dpdata.LabeledSystem(frame, fmt="abacus/scf")
        density_file_path = os.path.abspath(os.path.join(frame, "OUT.ABACUS", "ABACUS-CHARGE-DENSITY.restart"))
        s.data["density"] = np.array([[density_file_path]])
        ms.append(s)
    
    # Convert to deepmd format
    ms.to_deepmd_npy(output_path)
    
    # Make the dataset portable
    if portable:
        make_portable(output_path)

def make_portable(path: str):
    """
    Make the dataset portable by copying density files and updating references.
    """
    for f in glob.glob(f"{path}/**/density.npy", recursive=True):
        density = np.load(f)
        density_dir = os.path.join(os.path.dirname(f), "densities")
        os.makedirs(density_dir, exist_ok=True)
        new_density = []
        for i, density_file in enumerate(density):
            filename = f"ABACUS-CHARGE-DENSITY.{i:06d}.restart"
            os.system(f"cp {density_file[0]} {os.path.join(density_dir, filename)}")
            new_density.append([os.path.join("densities", filename)])
        np.save(f, new_density)

if __name__ == "__main__":
    sys_path = "/root/abacus_stru/binary/Mg0Al13Cu3"
    frame_paths = [os.path.join(sys_path, d) for d in os.listdir(sys_path) if os.path.isdir(os.path.join(sys_path, d))]
    output_path = "/root/dpmdata"
    process_density_dataset(frame_paths, output_path, True)
