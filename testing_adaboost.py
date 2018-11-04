import ADABoost
import numpy as np


np.random.seed(100)
sample_array_1 = np.reshape(np.random.randint(0, 10, 20), (5,2,2))
sample_array_2 = np.random.randint(0, 10, 5)
sample_distribution = np.array([.2] * 5)

def test_sample_creator_1():
    sample = ADABoost.sample_creator(sample_array_1, sample_array_2,
                sample_distribution, 7)
    assert np.shape(sample[0]) == (7,2,2)
