import numpy as np
from Dennis.worm.worm_utils import plot_worm

# in_out_data = [
#     ["posture/posture__mb01_053_N2tmd13.npy", "animations/worm_animation__mb01_053_N2tmd13.gif", "worm_posture"],
# ]

# name = "mb01_051_N2tmd9"
# name = "mb01_052_N2tmd10"
name = "mb01_053_N2tmd13"

in_out_data = [
    [f"nonan_posture/nonan_posture__{name}.npy",
     f"animations/nonan_worm_animation__{name}.gif",
     f"{name} (no nan)"],
]

for data_to_load, save_name, title in in_out_data:

    data = np.load(data_to_load)
    num_frames = 1000
    # num_frames = data.shape[0]
    plot_worm.animate_and_save(data,
                               file_name=save_name,
                               num_frames=num_frames,
                               interval=1000 / 5,
                               fps=10,
                               title=title)
