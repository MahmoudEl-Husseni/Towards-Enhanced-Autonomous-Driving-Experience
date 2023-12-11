import numpy as np
from matplotlib import pyplot as plt


def plotGT_trajectory(data, show=True, save=False, save_path=None):
    y = data["gt_marginal"]
    is_available = data["future_val_marginal"]
    V = data["vector_data"]

    X, idx = V[:, :44], V[:, 44].flatten()

    plt.figure(figsize=(15, 15), dpi=80)
    for i in np.unique(idx):
        _X = X[idx == i]
        if _X[:, 5:12].sum() > 0:
            plt.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
        else:
            plt.plot(_X[:, 0], _X[:, 1], color="black")
        plt.xlim([-224 // 4, 224 // 4])
        plt.ylim([-224 // 4, 224 // 4])
    
    plt.plot(
        y[is_available > 0][::10, 0],
        y[is_available > 0][::10, 1],
        "-o",
        label="gt",
    )
    plt.legend()

    if show:
        plt.show()

    if save:
        plt.savefig(save_path)
    
    
def plot_pred(data, 
              logits, 
              confidences, 
              ax=None, 
              use_top1=True, 
              trajectory_colors=None, 
              show=True,
              save=False,
              save_path=None):
    
    '''
    Plots the predicted trajectories for the given data.

    Args:
        data: A dictionary containing the following keys:
            "gt_marginal": The ground truth marginal distribution of the future states.
            "future_val_marginal": The marginal distribution of the future states predicted by the model.
            "vector_data": The vectorized data representing the past states and actions.
        logits: A tensor of shape (num_trajectories, 80, 2) containing the predicted logits for each trajectory and time step.
        confidences: A tensor of shape (num_trajectories,) containing the confidence scores for each trajectory.
        use_top1: Whether to only plot the top 1 predicted trajectory.
        trajectory_colors: A list of colors to use for plotting the predicted trajectories.

    Returns:
        None.
    '''

    if trajectory_colors is None:
        trajectory_colors = ["red", "green", "blue", "orange", "purple", "yellow"]

    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 15), dpi=80)

    y = data["gt_marginal"].reshape(80, 2)
    is_available = data["future_val_marginal"]
    V = data["vector_data"]

    X, idx = V[:, :44], V[:, 44].flatten()

    for i in np.unique(idx):
        _X = X[idx == i]
        if _X[:, 5:12].sum() > 0:
            ax.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
        else:
            ax.plot(_X[:, 0], _X[:, 1], color="black")
        ax.set_xlim([-224 // 4, 224 // 4])
        ax.set_ylim([-224 // 4, 224 // 4])

    ax.plot(
        y[is_available > 0][::10, 0],
        y[is_available > 0][::10, 1],
        "-o",
        label="gt",
    )

    argmax = confidences.argmax()
    ax.plot(
        logits[argmax][is_available > 0][::10, 0],
        logits[argmax][is_available > 0][::10, 1],
        "-o",
        label="pred top 1",
        color=trajectory_colors[argmax]
    )
    if not use_top1:
        for traj_id in range(len(logits)):
            if traj_id == argmax:
                continue

            alpha = confidences[traj_id].item()
            ax.plot(
                logits[traj_id][is_available > 0][::10, 0],
                logits[traj_id][is_available > 0][::10, 1],
                "-o",
                label=f"pred {traj_id} {alpha:.3f}",
                alpha=alpha,
                color=trajectory_colors[traj_id]
            )
    ax.legend()

    if show:
        plt.show()

    if save:
        fig.savefig(save_path)