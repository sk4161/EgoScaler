import numpy as np

def preprocess_traj(traj: np.ndarray, num_steps: int, return_padding_mask: bool = False):
    """
    Downsample the trajectory to the specified number of steps.
    If T < num_steps, pad the trajectory with the last point.

    Parameters:
    - traj (np.ndarray): Original trajectory with shape (T, D).
    - num_steps (int): Number of timesteps after downsampling.
    - return_padding_mask (bool): Whether to return the padding mask.

    Returns:
    - np.ndarray: Downsampled and padded trajectory with shape (num_steps, D).
    - (optional) np.ndarray: Padding mask with shape (num_steps,).
    """
    T, D = traj.shape

    if T >= num_steps:
        # Calculate indices for evenly spaced sampling
        indices = np.linspace(0, T - 1, num_steps).astype(int)
        # Sample the trajectory using the calculated indices
        sampled_traj = traj[indices]
        padding_mask = np.ones(num_steps, dtype=int)  # No padding
    else:
        # No downsampling needed, padding is required
        sampled_traj = traj.copy()
        # Calculate the number of padding steps
        pad_length = num_steps - T
        # Pad the trajectory with the last point
        pad = np.tile(traj[-1], (pad_length, 1))
        # Add padding to the trajectory
        sampled_traj = np.vstack([sampled_traj, pad])
        # Create the padding mask
        padding_mask = np.concatenate([np.ones(T, dtype=int), np.zeros(pad_length, dtype=int)])

    if return_padding_mask:
        return sampled_traj, padding_mask
    return sampled_traj

def smoothing_traj(traj: np.ndarray) -> np.ndarray:
    """
    Smooth the trajectory by averaging positions over neighboring frames.

    Parameters:
    - traj (np.ndarray): Original trajectory with shape (T, D).

    Returns:
    - np.ndarray: Smoothed trajectory with shape (T, D).
    """
    pos_seq = traj[:, :3]
    new_pos_seq = []

    for j in range(pos_seq.shape[0]):
        # Calculate the average for the j-th frame
        if j == 0:
            # First frame: Weight 3 for the current frame, 1 for the next two frames
            if pos_seq.shape[0] >= 3:
                mean = (3 * pos_seq[j] + pos_seq[j + 1] + pos_seq[j + 2]) / 5
            elif pos_seq.shape[0] == 2:
                mean = (3 * pos_seq[j] + pos_seq[j + 1]) / 4
            else:
                mean = pos_seq[j]
        elif j == 1:
            # Second frame: Weight 2 for the previous frame, 1 for the current and next two frames
            if pos_seq.shape[0] >= 4:
                mean = (2 * pos_seq[j - 1] + pos_seq[j] + pos_seq[j + 1] + pos_seq[j + 2]) / 5
            elif pos_seq.shape[0] == 3:
                mean = (2 * pos_seq[j - 1] + pos_seq[j] + pos_seq[j + 1]) / 4
            else:
                mean = pos_seq[j]
        elif j == pos_seq.shape[0] - 2:
            # Second-to-last frame: Use j-2, j-1, j, j+1
            if pos_seq.shape[0] >= 4:
                mean = (pos_seq[j - 2] + pos_seq[j - 1] + pos_seq[j] + pos_seq[j + 1]) / 4
            elif pos_seq.shape[0] == 3:
                mean = (pos_seq[j - 1] + pos_seq[j] + pos_seq[j + 1]) / 3
            else:
                mean = pos_seq[j]
        elif j == pos_seq.shape[0] - 1:
            # Last frame: Use j-2, j-1, j
            if pos_seq.shape[0] >= 3:
                mean = (pos_seq[j - 2] + pos_seq[j - 1] + pos_seq[j]) / 3
            elif pos_seq.shape[0] == 2:
                mean = (pos_seq[j - 1] + pos_seq[j]) / 2
            else:
                mean = pos_seq[j]
        else:
            # General case: Average over 5 frames
            mean = (pos_seq[j - 2] + pos_seq[j - 1] + pos_seq[j] + pos_seq[j + 1] + pos_seq[j + 2]) / 5

        new_pos_seq.append(mean)

    pos_seq = np.array(new_pos_seq)
    traj = np.concatenate([pos_seq, traj[:, 3:]], axis=-1)

    return traj