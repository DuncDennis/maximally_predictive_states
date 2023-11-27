import numpy as np
import sklearn


def get_neighbour_label(
        base_label: int,
        prev_label: int,
        model: sklearn.cluster.KMeans or sklearn.cluster.MiniBatchKMeans,
        return_convergence: bool = False) -> int | tuple[int, list]:
    """Get the label of the nearest neighbour in the direction of the prev_label.

    """
    cluster_centers = model.cluster_centers_

    base_center = cluster_centers[base_label]
    prev_center = cluster_centers[prev_label]

    if return_convergence:
        label_convergence = [prev_label]
    # print(f"--- {base_label, prev_label}")
    i = 0
    while True:
        # print(f"{i=}")
        i += 1
        half_point = (base_center + prev_center) * 0.5
        half_point_label = model.predict(half_point[np.newaxis, :])[0]
        if (half_point_label == base_label) or (i == 1000):
            if return_convergence:
                return prev_label, label_convergence
            else:
                return prev_label
        else:
            prev_label = half_point_label
            prev_center = cluster_centers[prev_label]

            if return_convergence:
                label_convergence.append(prev_label)


def lle_from_transition_matrix(
        msm_simulate_func,
        kmeans_model: sklearn.cluster.KMeans or sklearn.cluster.MiniBatchKMeans,
        start_label_base: int,
        start_label_perturbed: int,
        steps: int = int(1e3),
        part_time_steps: int = 15,
        steps_skip: int = 50,
        dt: float = 1.0,
        return_convergence: bool = False,
        seed: int = None,
) -> float | np.ndarray:
    cluster_centers = kmeans_model.cluster_centers_
    base_center = cluster_centers[start_label_base]
    start_label_perturbed = get_neighbour_label(start_label_base,
                                                start_label_perturbed,
                                                model=kmeans_model,
                                                return_convergence=False)

    pert_center = cluster_centers[start_label_perturbed]

    # log_divergence = np.zeros(steps)
    log_divergence = []

    x = base_center
    x_pert = pert_center
    x_label = start_label_base
    x_pert_label = start_label_perturbed

    initial_distances = []
    final_distances = []
    skipped_steps = 0
    for i_n in range(steps + steps_skip):
        print(f"{i_n=}")
        prev_norm_dx = np.linalg.norm(x_pert - x)
        initial_distances.append(prev_norm_dx)
        x_label = msm_simulate_func(n_steps=part_time_steps,
                                    start=x_label,
                                    dt=1,
                                    seed=seed)[-1]
        x_pert_label = msm_simulate_func(n_steps=part_time_steps,
                                         start=x_pert_label,
                                         dt=1,
                                         seed=seed)[-1]

        if x_label == x_pert_label:
            skipped_steps += 1
            continue

        x = cluster_centers[x_label]
        x_pert = cluster_centers[x_pert_label]

        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        final_distances.append(norm_dx)
        x_pert_label = get_neighbour_label(x_label,
                                           x_pert_label,
                                           model=kmeans_model,
                                           return_convergence=False)

        if i_n >= steps_skip:
            log_divergence.append(np.log(norm_dx / prev_norm_dx))
            # log_divergence[i_n - steps_skip] = np.log(norm_dx / prev_norm_dx)
    print(f"{skipped_steps}/{steps} skipped steps. ")
    print(f"{np.average(initial_distances)=}")
    print(f"{np.average(final_distances)=}")
    log_divergence = np.array(log_divergence)
    if return_convergence:
        return np.array(
            np.cumsum(log_divergence) / (np.arange(1, log_divergence.size + 1) * dt * part_time_steps)
        )
        # return np.array(
        #     np.cumsum(log_divergence) / (np.arange(1, steps + 1) * dt * part_time_steps)
        # )
    else:
        return float(np.average(log_divergence) / (dt * part_time_steps))
