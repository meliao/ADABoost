import numpy as np
def adaboost(weak_learner, m, T, data, labels):
    '''
    An implementation of the ADABoost algorithm as taught by Prof. Nati Srebro.

    ADABoost takes a weak learning algorithm and boosts its loss
    to arbitrarily-low loss.
    Inputs:
        weak_learner: A function which takes <m> samples and returns a
            hypothesis (a ML model)
        m: A integer corresponding to the sample size accepted by weak_learner
        T: The number of rounds of boosting to be performed
        data: The source of data on which the boosting algorithm learns
            \\WHAT IS THE DATATYPE??
        labels: A list of labels corresponding to the data
    '''

    #Initialization
    assert len(data) == len(labels)
    source_size = len(data)
    distribution = [1/source_size]*source_size
    weak_learners_lst = []
    alpha_lst = []
    epsilon_lst = []

    for i in range(T):
        continue
        #Create new sample object accroding to distribution
        (sample_data, sample_labels) = sample_creator(data, labels, distribution, m)

        new_learner = weak_learner(sample_data, sample_labels)
        (epsilon, delta) = evaluate(weak_learner)

    return

def sample_creator(data, labels, distribution, m):
    '''
    Creates a new sample of size <m> by randomly sampling from <distribution>,
    a discrete probability distribution over the cases in <data x labels>
    Inputs:
        data: the whole dataset
        labels: the whole labelset
        distribution: a list of floats of length equal to that of data and labels
        m: size of the sample to be created
    Returns:
        sample_data: a list of length m of data
        sample_labels: a list of length m of labels
    '''
    indices = np.arange(len(data))
    sample_indices = np.random.choice(indices, m, True, distribution)
    data_sample = data[sample_indices]
    label_sample = labels[indices]


    return (data_sample, label_sample)
