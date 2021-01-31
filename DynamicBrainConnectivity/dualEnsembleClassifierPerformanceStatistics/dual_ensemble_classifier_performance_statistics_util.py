import numpy as np


# generate a distribution based on mean and standard deviation
def generate_distribution(desired_std_dev, desired_mean, num_samples = 1000):
    # generate a distribution with (approximately) the desired std
    samples = np.random.normal(loc = 0.0, scale = desired_std_dev, size = num_samples)

    # find actual mean
    actual_mean = np.mean(samples)

    # transform it in a zero mean distribution
    zero_mean_samples = samples - (actual_mean)

    # compute the std of the zero mean distribution
    zero_mean_std = np.std(zero_mean_samples)

    # scale the distribution to the actual desired std
    scaled_samples = zero_mean_samples * (desired_std_dev / zero_mean_std)

    # make the distribution have the desired mean
    final_samples = scaled_samples + desired_mean

    return final_samples
