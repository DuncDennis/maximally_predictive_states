import numpy as np
from Dennis.larvee.larvee_utils import plot_larvee

# in_out_data = [
#     ["combined/video2.npy", "animations/900_frames/video2.gif", "vid2 raw"],
#     ["combined/video3.npy", "animations/900_frames/video3.gif", "vid3 raw"],
#     ["centered/video2_centered.npy", "animations/900_frames/video2_centered.gif", "vid2 centered"],
#     ["centered/video3_centered.npy", "animations/900_frames/video3_centered.gif", "vid3 centered"],
#     ["rotated/video2_rotated.npy", "animations/900_frames/video2_rotated.gif", "vid2 rotated"],
#     ["rotated/video3_rotated.npy", "animations/900_frames/video3_rotated.gif", "vid3 rotated"],
# ]

in_out_data = [
    ["combined/video2.npy", "animations/all_frames/video2.gif", "vid2 raw"],
    ["combined/video3.npy", "animations/all_frames/video3.gif", "vid3 raw"],
    ["centered/video2_centered.npy", "animations/all_frames/video2_centered.gif", "vid2 centered"],
    ["centered/video3_centered.npy", "animations/all_frames/video3_centered.gif", "vid3 centered"],
    ["rotated/video2_rotated.npy", "animations/all_frames/video2_rotated.gif", "vid2 rotated"],
    ["rotated/video3_rotated.npy", "animations/all_frames/video3_rotated.gif", "vid3 rotated"],
]

for data_to_load, save_name, title in in_out_data:

    data = np.load(data_to_load)

    plot_larvee.animate_and_save(data,
                                 file_name=save_name,
                                 num_frames=data.shape[0],
                                 interval=1000 / 5,
                                 fps=10,
                                 title=title)

