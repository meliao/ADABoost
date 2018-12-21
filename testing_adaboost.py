import ADABoost
import numpy as np


np.random.seed(100)
sample_data_1 = np.reshape(np.random.randint(0, 10, 20), (5,2,2))
sample_labels_1 = np.random.randint(0, 10, 5)
sample_distribution_1 = np.array([.2] * 5)

def test_sample_creator_1():
    sample = ADABoost.sample_creator(sample_distribution_1, 7)
    assert np.shape(sample) == (7,)

def test_sample_creator_2():
    sample = ADABoost.sample_creator(sample_distribution_1, 3)
    assert np.shape(sample) == (3,)

sample_distribution_2 = np.array([1,0,0,0,0,0,0])
def test_sample_creator_3():
    sample = ADABoost.sample_creator(sample_distribution_2, 4)
    np.testing.assert_array_equal(sample, [0,0,0,0])
