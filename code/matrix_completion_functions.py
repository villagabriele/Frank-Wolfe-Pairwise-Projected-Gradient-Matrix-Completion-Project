import numpy as np
import pandas as pd
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from tqdm import trange

def calculate_density(matrix):
    observed = np.count_nonzero(matrix)
    total = matrix.size
    return observed / total

def masking(M, mask):
    return M * mask

def split_data(M_obs, mask, validation_fraction=0.2, test_fraction=0.1):
    """
    Split observed entries into training, validation, and test masks.
    """
    
    # Get known entry indices
    known_indices = np.argwhere(mask)
    n_total_known = len(known_indices)

    # Number of test entries
    n_test = int(test_fraction * n_total_known)

    # Randomly select test indices
    test_indices = known_indices[np.random.choice(n_total_known, n_test, replace=False)]

    # Mask for test entries
    test_mask = np.zeros_like(M_obs, dtype=bool)
    for idx in test_indices:
        test_mask[tuple(idx)] = True

    # Remaining known indices (for train + validation)
    remaining_indices = np.array([idx for idx in known_indices if not test_mask[tuple(idx)]])

    # Number of validation entries (from remaining)
    n_validation = int(validation_fraction * len(remaining_indices))

    # Randomly select validation indices
    validation_indices = remaining_indices[np.random.choice(len(remaining_indices), n_validation, replace=False)]

    # Mask for validation entries
    validation_mask = np.zeros_like(M_obs, dtype=bool)
    for idx in validation_indices:
        validation_mask[tuple(idx)] = True

    # Training mask: known entries not in test or validation
    train_mask = mask & (~test_mask) & (~validation_mask)

    return train_mask, validation_mask, test_mask

def compute_rmse(X_pred, mask, X_true):
    return np.sqrt(np.mean((X_pred[mask == 1] - X_true[mask == 1]) ** 2))

def compute_gradient(X, M_obs, mask):
    return 2 * masking(X - M_obs, mask)

def linear_minimization_oracle(grad, delta):
    """
    Solve the linear minimization oracle (LMO) for the nuclear norm ball constraint.

    Parameters:
    - grad: gradient matrix
    - delta: radius of the nuclear norm ball (constraint on the sum of singular values)

    Returns:
    - rank_one_update: optimal rank-one matrix minimizing the inner product with the gradient, scaled by delta
    """
    u, s, vt = svds(-grad, k=1, solver="arpack")  # compute the leading singular vectors
    rank_one = np.outer(u[:, 0], vt[0, :])        # form the rank-one matrix
    return delta * rank_one

def away_oracle(grad, S):
    """
    Select the atom in the active set that maximizes the directional derivative (away step oracle).

    Parameters:
    - grad: gradient matrix
    - S: list of atoms currently in the active set

    Returns:
    - id_max: index of the atom in S with the largest inner product with the gradient
    """
    dot_products = [np.vdot(grad, S_k) for S_k in S]
    id_max = np.argmax(dot_products)
    return id_max

def duality_gap(X, S, grad):
    return np.vdot(X - S, grad)

def initialize_matrix(matrix, delta, rank=1):
    """
    Initialize a low-rank matrix with random values, normalized and scaled by delta.
    """
    np.random.seed(120)  
    U = np.random.rand(matrix.shape[0], rank)
    U_norm = U / np.linalg.norm(U, axis=0)  # Normalize U
    V = np.random.rand(matrix.shape[1], rank)
    V_norm = V / np.linalg.norm(V, axis=0)  # Normalize V
    init_mat = np.dot(U_norm, V_norm.T)

    return delta * init_mat

def optimal_alpha_line_search(grad, d_k, alpha_max):

    numerator = np.sum(grad/2 * d_k)
    denominator = np.sum(d_k * d_k)

    if denominator == 0:
        return 0.0

    alpha_star = - numerator / denominator
    alpha_opt = max(0.0, min(alpha_star, alpha_max))

    return alpha_opt

def alpha_armijo(grad, d_k, x_k, M_obs, mask, alpha_max):
    
    delta = 0.5
    gamma = 0.001
    alpha = alpha_max
    trace_dot_gamma = np.trace(grad.T @ d_k) * gamma  # trace is the inner product
    diff = x_k - M_obs
    obj_function = norm(masking(diff, mask))**2

    while norm(masking(diff + alpha * d_k, mask))**2 > obj_function + alpha * trace_dot_gamma:
        alpha *= delta

    return alpha

def alpha_lipschitz(grad, d_k, L, alpha_max):
    return min(-np.trace(grad.T @ d_k) / (L * np.linalg.norm(d_k)**2), alpha_max)

def plot_all_algorithms(results_fw, results_pair, results_pg, log_scale=False, title_prefix=""):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LOSS PLOT
    for step_rule, res in results_fw.items():
        if 'losses' in res and len(res['losses']) > 0:
            axes[0].plot(res['losses'], label=f"FW - {step_rule}")

    for step_rule, res in results_pair.items():
        if 'losses' in res and len(res['losses']) > 0:
            axes[0].plot(res['losses'], label=f"PairFW - {step_rule}")

    for step_rule, res in results_pg.items():
        if 'losses' in res and len(res['losses']) > 0:
            axes[0].plot(res['losses'], label=f"PG - {step_rule}")

    axes[0].set_title(f"{title_prefix} Loss per Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    if log_scale:
        axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True)

    # DUALITY GAP PLOT
    has_gap = False
    for step_rule, res in results_fw.items():
        if 'duality_gaps' in res and len(res['duality_gaps']) > 0:
            axes[1].plot(res['duality_gaps'], label=f"FW - {step_rule}")
            has_gap = True

    for step_rule, res in results_pair.items():
        if 'duality_gaps' in res and len(res['duality_gaps']) > 0:
            axes[1].plot(res['duality_gaps'], label=f"PairFW - {step_rule}")
            has_gap = True

    if has_gap:
        axes[1].set_title(f"{title_prefix} Duality Gap per Iteration")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Duality Gap")
        if log_scale:
            axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No Duality Gap data", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.show()

def FW(U_obs, delta, K, step_rule, eps, mask, verbose=1):
    """
    Parameters:
    - U_obs: observed matrix
    - delta: radius of the nuclear norm ball constraint
    - K: maximum number of iterations
    - step_rule: step size selection rule; one of ['dim', 'line', 'armijo', 'lipschitz']
    - eps: tolerance for the duality gap stopping criterion
    - mask: boolean or binary mask indicating observed entries
    - verbose: verbosity level (0 = silent, 1 = progress bar, 2 = detailed iteration info)

    Returns:
    - X_k: final estimate matrix after optimization
    - losses: list of loss values per iteration
    - duality_gaps: list of duality gap values per iteration
    """
    X_k = initialize_matrix(U_obs, delta, 1)
    losses = []
    duality_gaps = []

    # Choose iterator type based on verbose
    if verbose == 1:
        iterator = trange(K, desc="FW Iterations", leave=True)
    else:
        iterator = range(K)

    for k in iterator:
        grad = compute_gradient(X_k, U_obs, mask)
        X_k_hat = linear_minimization_oracle(grad, delta)

        duality_gap_k = duality_gap(X_k, X_k_hat, grad)
        if duality_gap_k < eps:
            if verbose == 2:
                print(f"Stopping at iteration {k} — duality gap {duality_gap_k:.4e} < eps {eps}")
            break

        d_k = X_k_hat - X_k

        # Step size selection
        if step_rule == 'dim':
            alpha_k = 2 / (k + 2)
        elif step_rule == 'line':
            alpha_k = optimal_alpha_line_search(grad, d_k, 1)
        elif step_rule == 'armijo':
            alpha_k = alpha_armijo(grad, d_k, X_k, U_obs, mask, 1)
        elif step_rule == 'lipschitz':
            alpha_k = alpha_lipschitz(grad, d_k, 2, 1)
        else:
            raise ValueError("Unknown step rule.")

        X_k += alpha_k * d_k

        loss = norm(masking(X_k - U_obs, mask))**2
        losses.append(loss)
        duality_gaps.append(duality_gap_k)

        if verbose == 2:
            print(f"Iteration {k}, Duality gap: {duality_gap_k:.4e}, Loss: {loss:.4f}")

    return X_k, losses, duality_gaps

def in_set(X, S):
    """
    Check if a given matrix (atom) X is present in the set S.
    """
    for i, A in enumerate(S):
        if np.array_equal(X, A):
            return i
    return -1  # not found

def pairwise_FW(U_obs, delta, K, step_rule, eps, mask, verbose=1):
    """
    Parameters:
    - U_obs: observed matrix
    - delta: radius of the nuclear norm ball constraint
    - K: maximum number of iterations
    - step_rule: step size selection rule; one of ['dim', 'line', 'armijo', 'lipschitz']
    - eps: tolerance for the duality gap stopping criterion
    - mask: boolean or binary mask indicating observed entries
    - verbose: verbosity level (0 = silent, 1 = progress bar, 2 = detailed iteration info)

    Returns:
    - X_k: final estimate matrix after optimization
    - losses: list of loss values per iteration
    - duality_gaps: list of duality gap values per iteration
    """
    X_k = initialize_matrix(U_obs, delta, 1)
    losses = []
    duality_gaps = []

    S = [X_k.copy()]      # list of atoms
    lambdas = [1.0]       # list of corresponding coefficients

    # Choose iterator type based on verbose
    if verbose == 1:
        iterator = trange(K, desc="Pairwise FW Iterations", leave=True)
    else:
        iterator = range(K)

    for k in iterator:
        grad = compute_gradient(X_k, U_obs, mask)
        X_k_FW = linear_minimization_oracle(grad, delta)

        duality_gap_k = duality_gap(X_k, X_k_FW, grad)

        if duality_gap_k < eps:
            if verbose == 2:
                print(f"Stopping at iteration {k} — duality gap {duality_gap_k:.4e} < eps {eps}")
            break

        # Pairwise step
        id_away_atom = away_oracle(grad, S)
        X_k_AS = S[id_away_atom]
        lambda_away_atom = lambdas[id_away_atom]

        d_k_FW = X_k_FW - X_k
        d_k_AS = X_k - X_k_AS
        d_k = d_k_FW + d_k_AS

        alpha_max = lambda_away_atom

        # Step size selection
        if step_rule == 'dim':
            alpha_k = min(2 / (k + 2), alpha_max)
        elif step_rule == 'line':
            alpha_k = optimal_alpha_line_search(grad, d_k, alpha_max)
        elif step_rule == 'armijo':
            alpha_k = alpha_armijo(grad, d_k, X_k, U_obs, mask, alpha_max)
        elif step_rule == 'lipschitz':
            alpha_k = alpha_lipschitz(grad, d_k, 2, alpha_max)
        else:
            raise ValueError("Unknown step rule.")

        # Update lambdas
        s_index = in_set(X_k_FW, S)
        if s_index != -1:
            lambdas[s_index] += alpha_k
        else:
            S.append(X_k_FW.copy())
            lambdas.append(alpha_k)

        lambdas[id_away_atom] -= alpha_k

        if lambdas[id_away_atom] <= 1e-10:
            del S[id_away_atom]
            del lambdas[id_away_atom]

        X_k += alpha_k * d_k

        loss = norm(masking(X_k - U_obs, mask))**2
        losses.append(loss)
        duality_gaps.append(duality_gap_k)

        if verbose == 2:
            print(f"Iteration {k}, Duality gap: {duality_gap_k:.4e}, Loss: {loss:.4f}")

    return X_k, losses, duality_gaps

def projection(Y, delta):
    """
    Project a matrix Y onto the nuclear norm ball of radius delta.

    Parameters:
    - Y: input matrix to be projected
    - delta: radius of the nuclear norm ball

    Returns:
    - X_hat: the projection of Y onto the nuclear norm ball
    """
    u, s, vt = svd(Y, full_matrices=False)
    k_size = len(s)
    tau = 0

    if sum(s) <= delta:
        X_hat = Y
    else :
        for j in range(k_size):
            current_threshold = (sum(s[:j+1])- delta)/(j+1)
            if s[j] - current_threshold > 0:
                tau = current_threshold
            else :
                break
        sigma_new = [max(i - tau,0) for i in s]
        sigma_new = np.diag(sigma_new)
        X_hat = u @ sigma_new @ vt

    return X_hat
    
def projected_gradient(U_obs, delta, K, step_rule, eps, mask, verbose=1):
    """
    Parameters:
    - U_obs: observed matrix
    - delta: radius of the nuclear norm ball constraint
    - K: maximum number of iterations
    - step_rule: step size selection rule; one of ['dim', 'line', 'armijo', 'lipschitz']
    - eps: tolerance for stopping criterion based on relative change in solution
    - mask: boolean or binary mask indicating observed entries
    - verbose: verbosity level (0 = silent, 1 = progress bar, 2 = detailed iteration info)

    Returns:
    - X_k: final estimate matrix after optimization
    - losses: list of loss values per iteration
    - variations: list of relative changes in solution norm per iteration
    """
    X_k = initialize_matrix(U_obs, delta, 1)
    losses = []
    variations = []
    L = 2

    if verbose == 1:
        iterator = trange(K, desc="Projected Gradient Iterations", leave=True)
    else:
        iterator = range(K)

    for k in iterator:
        grad = compute_gradient(X_k, U_obs, mask)

        s_k = 1/L
        Y = X_k - s_k * grad
        X_k_hat = projection(Y, delta)

        #stopping condition
        norm_X_k = np.linalg.norm(X_k, 'fro')
        if norm_X_k > 1e-10:
            solution_change = np.linalg.norm(X_k_hat - X_k, 'fro') / norm_X_k
            variations.append(solution_change)
            if solution_change < eps:
                if verbose == 2:
                    print(f"Stopping at iteration {k} — Loss change {solution_change:.4e} < eps {eps}")
                break

        if step_rule == 'dim':
            alpha_k = 2 / (k + 2)
        elif step_rule == 'line':
            alpha_k = optimal_alpha_line_search(grad, X_k_hat - X_k, 1)
        elif step_rule == 'armijo':
            alpha_k = alpha_armijo(grad, X_k_hat - X_k, X_k, U_obs, mask, 1)
        elif step_rule == 'lipschitz':
            alpha_k = alpha_lipschitz(grad, X_k_hat - X_k, L, 1)

        X_k += alpha_k * (X_k_hat - X_k)

        loss = norm(masking(X_k - U_obs, mask))**2
        losses.append(loss)

        if verbose == 2:
            print(f"Iteration {k}, Loss change: {solution_change:.4e}, Loss: {loss:.4f}")

    return X_k, losses, variations

def load_netflix_data(filepath, max_lines=None):
    
    data = []
    movie_id = None
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line.endswith(':'):
                movie_id = int(line[:-1])
            else:
                user_id, rating, date = line.split(',')
                data.append((int(user_id), movie_id, int(rating), date))

    return pd.DataFrame(data, columns=["user_id", "movie_id", "rating", "date"])

def build_results_df(results_dict, method_name):
    
    rows = []
    for s in sorted(results_dict.keys()):
        entry = results_dict[s]
        rows.append({
            'method': method_name,
            'stepsize': s,
            'rmse_test': entry['rmse_test'],
            'rank': entry['rank'],
            'time_sec': entry['time']
        })
    return pd.DataFrame(rows)
