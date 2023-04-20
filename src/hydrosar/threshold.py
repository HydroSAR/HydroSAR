# Turned off flake8 because we haven't refactored 3rd party provided functions
# flake8: noqa
import numpy as np


def _make_histogram(image):
    image = image.flatten()
    indices = np.nonzero(np.isnan(image))
    image[indices] = 0
    indices = np.nonzero(np.isinf(image))
    image[indices] = 0
    del indices
    size = image.size
    maximum = int(np.ceil(np.amax(image)) + 1)
    histogram = np.zeros((1, maximum))
    for i in range(0, size):
        floor_value = np.floor(image[i]).astype(np.uint8)
        if 0 < floor_value < maximum - 1:
            temp1 = image[i] - floor_value
            temp2 = 1 - temp1
            histogram[0, floor_value] = histogram[0, floor_value] + temp1
            histogram[0, floor_value - 1] = histogram[0, floor_value - 1] + temp2
    histogram = np.convolve(histogram[0], [1, 2, 3, 2, 1])
    histogram = histogram[2:(histogram.size - 3)]
    histogram /= np.sum(histogram)
    return histogram


def _make_distribution(m, v, g, x):
    x = x.flatten()
    m = m.flatten()
    v = v.flatten()
    g = g.flatten()
    y = np.zeros((len(x), m.shape[0]))
    for i in range(0, m.shape[0]):
        d = x - m[i]
        amp = g[i] / np.sqrt(2 * np.pi * v[i])
        y[:, i] = amp * np.exp(-0.5 * (d * d) / v[i])
    return y


def expectation_maximization_threshold(tile: np.ndarray, number_of_classes: int = 3) -> float:
    """Water threshold Calculation using a multi-mode Expectation Maximization Approach

    Thresholding works best when backscatter tiles are provided on a decibel scale
    to get Gaussian distribution that is scaled to a range of 0-255, and performed
    on a small tile that is likely to have a transition between water and not water.


    Args:
        tile: array of backscatter values for a tile from an RTC raster
        number_of_classes: classify the tile into this many classes. Typically, three
            classes capture: (1) urban and bright slopes, (2) average brightness farmland,
            and (3) water, as is often seen in the US Midwest.

    Returns:
        threshold: threshold value that can be used to create a water extent map
    """

    image_copy = tile.copy()
    image_copy2 = np.ma.filled(tile.astype(float), np.nan)  # needed for valid posterior_lookup keys
    image = tile.flatten()
    minimum = np.amin(image)
    image = image - minimum + 1
    maximum = np.amax(image)

    histogram = _make_histogram(image)
    nonzero_indices = np.nonzero(histogram)[0]
    histogram = histogram[nonzero_indices]
    histogram = histogram.flatten()
    class_means = (
            (np.arange(number_of_classes) + 1) * maximum /
            (number_of_classes + 1)
    )
    class_variances = np.ones(number_of_classes) * maximum
    class_proportions = np.ones(number_of_classes) * 1 / number_of_classes
    sml = np.mean(np.diff(nonzero_indices)) / 1000
    iteration = 0
    while True:
        class_likelihood = _make_distribution(
            class_means, class_variances, class_proportions, nonzero_indices
        )
        sum_likelihood = np.sum(class_likelihood, 1) + np.finfo(
            class_likelihood[0][0]).eps
        log_likelihood = np.sum(histogram * np.log(sum_likelihood))
        for j in range(0, number_of_classes):
            class_posterior_probability = (
                    histogram * class_likelihood[:, j] / sum_likelihood
            )
            class_proportions[j] = np.sum(class_posterior_probability)
            class_means[j] = (
                    np.sum(nonzero_indices * class_posterior_probability)
                    / class_proportions[j]
            )
            vr = (nonzero_indices - class_means[j])
            class_variances[j] = (
                    np.sum(vr * vr * class_posterior_probability)
                    / class_proportions[j] + sml
            )
            del class_posterior_probability, vr
        class_proportions += 1e-3
        class_proportions /= np.sum(class_proportions)
        class_likelihood = _make_distribution(
            class_means, class_variances, class_proportions, nonzero_indices
        )
        sum_likelihood = np.sum(class_likelihood, 1) + np.finfo(
            class_likelihood[0, 0]).eps
        del class_likelihood
        new_log_likelihood = np.sum(histogram * np.log(sum_likelihood))
        del sum_likelihood
        if (new_log_likelihood - log_likelihood) < 0.000001:
            break
        iteration += 1
    del log_likelihood, new_log_likelihood
    class_means = class_means + minimum - 1
    s = image_copy.shape
    posterior = np.zeros((s[0], s[1], number_of_classes))
    posterior_lookup = dict()
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            pixel_val = image_copy2[i, j]
            if pixel_val in posterior_lookup:
                for n in range(0, number_of_classes):
                    posterior[i, j, n] = posterior_lookup[pixel_val][n]
            else:
                posterior_lookup.update({pixel_val: [0] * number_of_classes})
                for n in range(0, number_of_classes):
                    x = _make_distribution(
                        class_means[n], class_variances[n], class_proportions[n],
                        image_copy[i, j]
                    )
                    posterior[i, j, n] = x * class_proportions[n]
                    posterior_lookup[pixel_val][n] = posterior[i, j, n]

    sorti = np.argsort(class_means)
    xvec = np.arange(class_means[sorti[0]], class_means[sorti[1]], step=.05)
    x1 = _make_distribution(class_means[sorti[0]], class_variances[sorti[0]], class_proportions[sorti[0]], xvec)
    x2 = _make_distribution(class_means[sorti[1]], class_variances[sorti[1]], class_proportions[sorti[1]], xvec)
    dx = np.abs(x1 - x2)

    return xvec[np.argmin(dx)]
