import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


def plot_single_frame_data(frame_data: np.ndarray, ax: Axes = None):
    if ax is None:
        # Create a new figure and 2D axes
        fig, ax = plt.subplots()

    nr_skeleton_pnts = frame_data.shape[0]
    if nr_skeleton_pnts == 24:  # 3 tail points
        nr_tail = 3
        last_segment_to_head = [-4, 0]

    elif nr_skeleton_pnts == 22:  # 2 tail points.
        nr_tail = 1
        last_segment_to_head = [-2, 0]

    scatter_colors = ["red"] + ["black"] * 20 + [
        "yellow"] * nr_tail  # red: head, white: body, yellow: tail.
    ax.scatter(x=frame_data[:, 0], y=frame_data[:, 1], c=scatter_colors)
    ax.plot(frame_data[:-nr_tail, 0], frame_data[:-nr_tail, 1], c="black")  # body curve
    ax.plot(frame_data[last_segment_to_head, 0], frame_data[last_segment_to_head, 1],
            c="black")
    ax.plot(frame_data[[11, -1], 0], frame_data[[11, -1], 1], c="black")
    ax.plot(frame_data[[10, -1], 0], frame_data[[10, -1], 1], c="black")
    ax.set_aspect('equal')


def plot_larvee(frame: int, data: np.ndarray, ax: Axes = None, title: str = None):
    """Plot one frame of larvee data.


    """
    if ax is None:
        # Create a new figure and 2D axes
        fig, ax = plt.subplots()

    frame_data = data[frame, :, :]

    ax.clear()
    plot_single_frame_data(frame_data, ax=ax)
    ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
    ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
    if title is not None:
        ax.set_title(f"{title}, {frame=}")
    else:
        ax.set_title(f"{frame=}")


def animate_and_save(data: np.ndarray,
                     file_name: str,
                     num_frames: int = 100,
                     interval: float = 1000 / 5,
                     fps: int = 10,
                     title: str = None):
    fig, ax = plt.subplots()
    # Set any additional parameters for your plot or axes if needed

    # Create the animation
    animation = FuncAnimation(fig, plot_larvee, frames=num_frames, fargs=(data, ax, title),
                              interval=interval)

    # If you want to save the animation as a GIF, uncomment the following line
    animation.save(file_name, fps=fps)
