"""Unmodified functions extracted from coupledcriticality_figure_tools.py.
See data/manifest.json for source provenance. Only imports were minimized.
"""
import numpy as np
import powerlaw

def scalar(value) -> float:
    arr = np.asarray(value).ravel()
    return float(arr[0]) if len(arr) else np.nan


def linear_fit(x, y):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    fit = model.fit(np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1)))
    return scalar(fit.coef_), scalar(fit.intercept_)


def noise_relation(size, length):
    size = np.asarray(size)
    length = np.asarray(length)
    duration_list = np.unique(length)
    mean_size = []
    for duration in duration_list:
        mean_size.append(np.mean(size[np.where(length == duration)[0]]))
    mean_size = np.asarray(mean_size)
    keep = (duration_list > 0) & (mean_size > 0) & np.isfinite(mean_size)
    duration_list = duration_list[keep]
    mean_size = mean_size[keep]
    slope, bias = linear_fit(np.log10(duration_list), np.log10(mean_size))
    return slope, bias, duration_list, mean_size


def pdf_points(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) == 0:
        return np.array([]), np.array([])
    edges, prob = powerlaw.pdf(values, linear_bins=True)
    x = np.asarray(edges[1:])
    y = np.asarray(prob)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[keep], y[keep]


def loglog_pdf(ax, values, *, color, label=None, marker=".", linestyle="-", linewidth=1.8):
    x, y = pdf_points(values)
    if len(x):
        ax.loglog(x, y, marker=marker, linestyle=linestyle, color=color, label=label, linewidth=linewidth)


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_3d_axis(ax) -> None:
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)


def plot_noise_relation(ax, duration, size, *, line_slope=None, label_text=True, bias_offset=0.0):
    model_slope, bias, duration_list, mean_size = noise_relation(size, duration)
    slope_to_plot = model_slope if line_slope is None else scalar(line_slope)
    ax.loglog(duration_list, mean_size, ".", color="0.45", markersize=4)
    ax.plot(duration_list, np.exp(np.log(duration_list) * slope_to_plot + bias + bias_offset), "-", color="k", linewidth=1.5)
    if label_text:
        ax.text(
            0.05,
            0.10,
            rf"$1/\sigma\nu z={model_slope:.2f}$",
            transform=ax.transAxes,
            fontsize=10,
        )
    ax.set_xlabel("Duration")
    ax.set_ylabel(r"$<S>(T)$")
    clean_axis(ax)
    return model_slope


def meanfield_couple(sigma_e, sigma_i, LBI=1, GBE=2, N_total=2000):
    import sympy as sy

    tau_di = [12.5, 5.0]
    backgroundfre = 10
    tau_e, tau_i, tau_de, tau_r = 20, 10, 2, 0.5
    Ve_rest, Vi_rest, Ve_rev, Vi_rev, V_threshold = -70, -70, 0, -70, -50
    g_size = 1 / np.sqrt(N_total / 1000)

    g = np.array([0.012, 0.024, 0.18, 0.31, 0.022, 0.040]) * g_size
    gee, gie, gei, gii, geo, gio = g[0] * GBE / 2, g[1] * GBE / 2, g[2], g[3], g[4], g[5]
    gei = gei * LBI
    ge0e1, ge1e0, gi0e1, gi1e0 = gee, gee, gie, gie

    p_inter = np.array([[0.2, 0.2], [0.2, 0.2]])
    no = p_inter[0, 0] * 0.8 * N_total
    ne = p_inter[0, 0] * 0.8 * N_total
    ni = p_inter[0, 0] * 0.2 * N_total
    Qono = backgroundfre * no / (10**3)

    V_e0, V_i0 = sy.Symbol("x0"), sy.Symbol("y0")
    V_e1, V_i1 = sy.Symbol("x1"), sy.Symbol("y1")
    sigma_e0, sigma_e1 = sigma_e[0], sigma_e[1]
    sigma_i0, sigma_i1 = sigma_i[0], sigma_i[1]

    f0 = (Ve_rest - V_e0) / tau_e + (Ve_rev - V_e0) * (
        geo * Qono
        + gee * ne * (1 / (1 + sy.exp((V_threshold - V_e0) * np.pi / (sigma_e0 * np.sqrt(3)))))
        + ge0e1 * ne * (1 / (1 + sy.exp((V_threshold - V_e1) * np.pi / (sigma_e1 * np.sqrt(3)))))
    ) + (Vi_rev - V_e0) * gei * ni * (
        1 / (1 + sy.exp((V_threshold - V_i0) * np.pi / (sigma_i0 * np.sqrt(3))))
    )

    f1 = (Vi_rest - V_i0) / tau_i + (Ve_rev - V_i0) * (
        gio * Qono
        + gie * ne * (1 / (1 + sy.exp((V_threshold - V_e0) * np.pi / (sigma_e0 * np.sqrt(3)))))
        + gi0e1 * ne * (1 / (1 + sy.exp((V_threshold - V_e1) * np.pi / (sigma_e1 * np.sqrt(3)))))
    ) + (Vi_rev - V_i0) * gii * ni * (
        1 / (1 + sy.exp((V_threshold - V_i0) * np.pi / (sigma_i0 * np.sqrt(3))))
    )

    f2 = (Ve_rest - V_e1) / tau_e + (Ve_rev - V_e1) * (
        geo * Qono
        + gee * ne * (1 / (1 + sy.exp((V_threshold - V_e1) * np.pi / (sigma_e1 * np.sqrt(3)))))
        + ge1e0 * ne * (1 / (1 + sy.exp((V_threshold - V_e0) * np.pi / (sigma_e0 * np.sqrt(3)))))
    ) + (Vi_rev - V_e1) * gei * ni * (
        1 / (1 + sy.exp((V_threshold - V_i1) * np.pi / (sigma_i1 * np.sqrt(3))))
    )

    f3 = (Vi_rest - V_i1) / tau_i + (Ve_rev - V_i1) * (
        gio * Qono
        + gie * ne * (1 / (1 + sy.exp((V_threshold - V_e1) * np.pi / (sigma_e1 * np.sqrt(3)))))
        + gi1e0 * ne * (1 / (1 + sy.exp((V_threshold - V_e0) * np.pi / (sigma_e0 * np.sqrt(3)))))
    ) + (Vi_rev - V_i1) * gii * ni * (
        1 / (1 + sy.exp((V_threshold - V_i1) * np.pi / (sigma_i1 * np.sqrt(3))))
    )

    value, count = 1, 0
    while (value > 10 ** (-3)) and (count < 10):
        Ve0, Vi0, Ve1, Vi1 = sy.nsolve(
            [f0, f1, f2, f3],
            [V_e0, V_i0, V_e1, V_i1],
            [-50 - count, -50 - count, -50 - count, -50 - count],
            verify=False,
        )
        Ve0, Ve1, Vi0, Vi1 = float(Ve0), float(Ve1), float(Vi0), float(Vi1)
        value = abs(f0.evalf(subs={V_e0: Ve0, V_e1: Ve1, V_i0: Vi0, V_i1: Vi1}))
        value += abs(f1.evalf(subs={V_e0: Ve0, V_e1: Ve1, V_i0: Vi0, V_i1: Vi1}))
        value += abs(f2.evalf(subs={V_e0: Ve0, V_e1: Ve1, V_i0: Vi0, V_i1: Vi1}))
        value += abs(f3.evalf(subs={V_e0: Ve0, V_e1: Ve1, V_i0: Vi0, V_i1: Vi1}))
        count += 1

    h_e0 = np.exp((V_threshold - Ve0) * np.pi / (sigma_e0 * np.sqrt(3)))
    h_i0 = np.exp((V_threshold - Vi0) * np.pi / (sigma_i0 * np.sqrt(3)))
    h_e1 = np.exp((V_threshold - Ve1) * np.pi / (sigma_e1 * np.sqrt(3)))
    h_i1 = np.exp((V_threshold - Vi1) * np.pi / (sigma_i1 * np.sqrt(3)))

    dQedVe0 = (np.pi / (sigma_e0 * np.sqrt(3))) * (h_e0 / ((1 + h_e0) ** 2))
    dQidVi0 = (np.pi / (sigma_i0 * np.sqrt(3))) * (h_i0 / ((1 + h_i0) ** 2))
    dQedVe1 = (np.pi / (sigma_e1 * np.sqrt(3))) * (h_e1 / ((1 + h_e1) ** 2))
    dQidVi1 = (np.pi / (sigma_i1 * np.sqrt(3))) * (h_i1 / ((1 + h_i1) ** 2))

    tau_di0, tau_di1 = tau_di[0], tau_di[1]
    Jacobi0 = np.array(
        [
            [
                -1 / tau_e
                - geo * Qono
                - gee * ne * (1 / (1 + h_e0))
                - (ge0e1 * ne * (1 / (1 + h_e1)))
                - (gei * ni * (1 / (1 + h_i0))),
                0,
                (Ve_rev - Ve0) * gee,
                0,
                (Vi_rev - Ve0) * gei,
                0,
            ],
            [
                0,
                -1 / tau_i
                - gio * Qono
                - gie * ne * (1 / (1 + h_e0))
                - (gi0e1 * ne * (1 / (1 + h_e1)))
                - (gii * ni * (1 / (1 + h_i0))),
                (Ve_rev - Vi0) * gie,
                0,
                (Vi_rev - Vi0) * gii,
                0,
            ],
            [0, 0, 0, 1, 0, 0],
            [ne * dQedVe0 / (tau_de * tau_r), 0, -1 / (tau_de * tau_r), -(1 / tau_de + 1 / tau_r), 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, ni * dQidVi0 / (tau_di0 * tau_r), 0, 0, -1 / (tau_di0 * tau_r), -(1 / tau_di0 + 1 / tau_r)],
        ]
    )
    Jacobi1 = np.array(
        [
            [0, 0, (Ve_rev - Ve0) * ge0e1, 0, 0, 0],
            [0, 0, (Ve_rev - Vi0) * gi0e1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    Jacobi2 = np.array(
        [
            [0, 0, (Ve_rev - Ve1) * ge1e0, 0, 0, 0],
            [0, 0, (Ve_rev - Vi1) * gi1e0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    Jacobi3 = np.array(
        [
            [
                -1 / tau_e
                - geo * Qono
                - gee * ne * (1 / (1 + h_e1))
                - (ge1e0 * ne * (1 / (1 + h_e0)))
                - (gei * ni * (1 / (1 + h_i1))),
                0,
                (Ve_rev - Ve1) * gee,
                0,
                (Vi_rev - Ve1) * gei,
                0,
            ],
            [
                0,
                -1 / tau_i
                - gio * Qono
                - gie * ne * (1 / (1 + h_e1))
                - (gi1e0 * ne * (1 / (1 + h_e0)))
                - (gii * ni * (1 / (1 + h_i1))),
                (Ve_rev - Vi0) * gie,
                0,
                (Vi_rev - Vi0) * gii,
                0,
            ],
            [0, 0, 0, 1, 0, 0],
            [ne * dQedVe1 / (tau_de * tau_r), 0, -1 / (tau_de * tau_r), -(1 / tau_de + 1 / tau_r), 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, ni * dQidVi1 / (tau_di1 * tau_r), 0, 0, -1 / (tau_di1 * tau_r), -(1 / tau_di1 + 1 / tau_r)],
        ]
    )

    jacobi = np.vstack((np.hstack((Jacobi0, Jacobi1)), np.hstack((Jacobi2, Jacobi3))))
    eigvalue, _ = np.linalg.eig(jacobi)
    return float(np.mean(np.max(np.real(eigvalue))))


def low_dim_presentation_2026(neuronsets, neuronset_lengths=None, signal_num=None, ana_dim=3, random_state=0):
    """PCA view used by the 2026 analysis scripts.

    The source scripts standardize the binary avalanche vectors, run PCA, then
    choose the PC combination with the largest silhouette coefficient.
    """
    from itertools import combinations

    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    signal_num = signal_num or len(neuronsets)
    matrices = []
    labels = []
    lengths = []
    for signal_idx in range(signal_num):
        matrix = np.asarray(neuronsets[signal_idx])
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            lengths.append(0)
            continue
        if neuronset_lengths is not None:
            usable = min(int(neuronset_lengths[signal_idx]), matrix.shape[1])
            matrix = matrix[:, :usable]
        matrices.append(matrix.T)
        labels.extend([signal_idx] * matrix.shape[1])
        lengths.append(matrix.shape[1])

    if not matrices:
        raise ValueError("No nonzero avalanche representation data found.")

    data = np.vstack(matrices)
    labels = np.asarray(labels)
    data = StandardScaler().fit_transform(data)
    n_components = min(10, data.shape[0] - 1, data.shape[1])
    if n_components < ana_dim:
        raise ValueError(f"Need at least {ana_dim} PCA components, got {n_components}.")

    pca = PCA(n_components=n_components, random_state=random_state)
    data_pca = pca.fit_transform(data)

    best_dims = tuple(range(ana_dim))
    best_score = -np.inf
    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
        for dims in combinations(range(n_components), ana_dim):
            try:
                score = silhouette_score(data_pca[:, dims], labels, metric="euclidean")
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_dims = dims
    if not np.isfinite(best_score):
        best_score = np.nan

    colorlist0 = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:pink", "tab:gray", "tab:purple", "tab:brown"]
    colors = [colorlist0[label % len(colorlist0)] for label in labels]
    return data_pca[:, best_dims], colors, lengths, labels, best_dims, best_score


def neuronset_to_matrix(avalanche_neuronidx_list_signal, noiselen=4, use_lda=False):
    import sklearn.preprocessing

    neuronset_noise = None
    lenlist = []
    labels = []
    for noiseidx in range(noiselen):
        neuronsets = avalanche_neuronidx_list_signal[noiseidx]
        if len(neuronsets) == 0:
            lenlist.append(0 if neuronset_noise is None else neuronset_noise.shape[1])
            continue
        temporal = [len(np.where(np.sum(neuronset, axis=0) != 0)[0]) for neuronset in neuronsets]
        neuronset = neuronsets[int(np.argmax(temporal))]
        nonzero = np.where(np.sum(neuronset, axis=0) != 0)[0]
        selected = neuronset[:, nonzero]
        if neuronset_noise is None:
            neuronset_noise = selected
        else:
            neuronset_noise = np.hstack([neuronset_noise, selected])
        labels.extend([noiseidx] * selected.shape[1])
        lenlist.append(neuronset_noise.shape[1])
    if neuronset_noise is None or neuronset_noise.shape[1] == 0:
        raise ValueError("No nonzero avalanche representation data found.")
    X = sklearn.preprocessing.normalize(neuronset_noise.T)
    return X, np.asarray(labels), lenlist


def pca_projection_with_optional_illness_filter(avalanche_signal, noiselen=6, illness_dim=1, illness_threshold=0.08):
    import sklearn.preprocessing

    X, labels, lenlist = neuronset_to_matrix(avalanche_signal, noiselen=noiselen)
    covdata = np.dot(X.T, X)
    _, eigvec = np.linalg.eig(covdata)
    initial = np.real(np.dot(X, eigvec))

    keep = np.ones(initial.shape[0], dtype=bool)
    if illness_dim is not None:
        keep = initial[:, illness_dim] <= illness_threshold
    X_filtered = sklearn.preprocessing.normalize(X[keep])
    labels_filtered = labels[keep]
    covdata = np.dot(X_filtered.T, X_filtered)
    _, eigvec = np.linalg.eig(covdata)
    lddata = np.real(np.dot(X_filtered, eigvec))

    colorlist0 = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:pink", "tab:gray", "tab:purple", "tab:brown"]
    colors = [colorlist0[label % len(colorlist0)] for label in labels_filtered]
    return lddata, colors, lenlist, labels_filtered


def plot_3d_points(ax, data, colors, dims=(0, 1, 2), labels=("PC1", "PC2", "PC3"), s=8, alpha=0.75):
    for point, color in zip(data, colors):
        ax.plot(point[dims[0]], point[dims[1]], point[dims[2]], ".", color=color, markersize=s, alpha=alpha)
    ax.set_xlabel(labels[0], labelpad=-2, fontsize=7)
    ax.set_ylabel(labels[1], labelpad=-2, fontsize=7)
    ax.set_zlabel(labels[2], labelpad=-2, fontsize=7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="major", labelsize=0, pad=-2)
    try:
        ax.set_box_aspect((1, 1, 0.82))
    except Exception:
        pass
    style_3d_axis(ax)


def activity_mask_2026(spiketime, spikeidx, min_time=0, max_time=250):
    spiketime = np.asarray(spiketime)
    spikeidx = np.asarray(spikeidx)
    mask = (spiketime >= min_time) & (spiketime <= max_time)
    return spiketime[mask], spikeidx[mask]


def avalanche_stats_from_spikes(spiketime, spikeindex, ava_thre=50, cutsize=300000):
    spiketime = np.asarray(spiketime)
    spikeindex = np.asarray(spikeindex)
    if len(spiketime) < 3:
        return np.array([]), [], [], np.array([])
    order = np.argsort(spiketime)
    spiketime = spiketime[order]
    spikeindex = spikeindex[order]
    isi_ave = float(np.mean(np.diff(spiketime)))
    if not np.isfinite(isi_ave) or isi_ave <= 0:
        return np.array([]), [], [], np.array([])

    size_bins, neuron_bins = [], []
    for cutindex in range(int(len(spiketime) / cutsize) + 1):
        beginindex = cutindex * cutsize
        endindex = min((cutindex + 1) * cutsize, len(spiketime))
        if endindex <= beginindex:
            continue
        times = list(spiketime[beginindex:endindex] - spiketime[beginindex])
        indexes = list(spikeindex[beginindex:endindex])
        if not times:
            continue
        for binidx in range(max(0, int(max(times) / isi_ave) - 1)):
            t_end = (binidx + 1) * isi_ave
            neurons = []
            while times and times[0] < t_end:
                neurons.append(indexes.pop(0))
                times.pop(0)
            size_bins.append(len(neurons))
            neuron_bins.append(neurons)

    ava_size_list, len_list, ava_idx_list, size_t = [], [], [], []
    cursor = 0
    for idx in range(len(size_bins)):
        if idx < cursor or size_bins[idx] == 0:
            continue
        length = 1
        while idx + length < len(size_bins) and size_bins[idx + length] != 0:
            length += 1
        len_list.append(length)
        ava_size_list.append(sum(size_bins[idx : idx + length]))
        ava_idx_list.append(np.hstack(neuron_bins[idx : idx + length]) if length else np.array([]))
        size_t.append(idx)
        cursor = idx + length

    if not ava_size_list:
        return np.array([]), [], [], np.array([])
    ava_size_list[0] = 0
    ava_t = (np.asarray(size_t) + np.asarray(len_list) / 2) * isi_ave
    large_idx = np.where(np.asarray(ava_size_list) > ava_thre)[0]
    return ava_t[large_idx], ava_idx_list, ava_size_list, ava_t


def figure4_rewiring_reliable_neuronset_2026(
    large_time_list,
    neuronidx_list,
    size_list,
    ava_t_list,
    netsize=1600,
    trial_max=100,
    ava_thre=20,
    attractor=40,
):
    timbinlist = [0, 1, 2, 3, 4]
    neuronset_tmp = []
    zero_count_tmp = []
    spatial_tmp = []

    for timebin in timbinlist:
        precise_time = attractor * 5 + timebin
        large_neuron_list = []
        for trial in range(trial_max):
            trial_sizes = np.asarray(size_list[trial])
            trial_times = np.asarray(ava_t_list[trial])
            large_idx = np.where(trial_sizes > ava_thre)[0]
            large_times = trial_times[large_idx] if len(large_idx) else np.array([])
            matches = np.where((large_times > precise_time - 1) & (large_times < precise_time + 4))[0]
            if len(matches) == 0:
                large_neuron_list.append(np.array([], dtype=int))
                continue
            match_sizes = np.array([len(neuronidx_list[trial][large_idx[match]]) for match in matches])
            selected = large_idx[matches[int(np.argmax(match_sizes))]]
            large_neuron_list.append(np.asarray(neuronidx_list[trial][selected], dtype=int))

        neuronset = np.zeros((netsize, trial_max))
        for trial, neurons in enumerate(large_neuron_list):
            neurons = neurons[(neurons >= 0) & (neurons < netsize)]
            neuronset[neurons, trial] = 1

        zero_trials = np.where(np.sum(neuronset, axis=0) == 0)[0]
        zero_count_tmp.append(len(zero_trials))
        nonzero = np.delete(neuronset, zero_trials, axis=1)
        if nonzero.shape[1] == 0:
            nonmax = nonzero
        else:
            fullactive = np.where(np.sum(nonzero, axis=0) == netsize)[0]
            nonmax = np.delete(nonzero, fullactive, axis=1)
        if len(zero_trials) == trial_max - 1 or nonmax.shape[1] <= 1:
            spatial_tmp.append(0)
        else:
            spatialcorr = np.corrcoef(nonmax.T)
            spatial_tmp.append((np.sum(spatialcorr) - len(spatialcorr)) / (len(spatialcorr) * (len(spatialcorr) - 1)))
        neuronset_tmp.append(neuronset)

    score_for_choice = np.asarray(zero_count_tmp, dtype=float) * np.asarray(spatial_tmp, dtype=float)
    best = int(np.argmin(score_for_choice))
    signal_reliable = neuronset_tmp[best]
    removezero = np.delete(signal_reliable, np.where(np.sum(signal_reliable, axis=0) == 0)[0], axis=1)
    removefull = np.delete(removezero, np.where(np.sum(removezero, axis=0) == netsize)[0], axis=1)
    return removefull
