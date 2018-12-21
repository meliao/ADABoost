import numpy as np
def adaboost(weak_learner, m, T, data, labels, call_1, call_2, args_1 = None, args_2 = None):
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
        call_1: function call for training the weak learner
        call_2: function call for using the weak learner for prediction
        args_1: (optional) additional arguments for call_1
        args_2: (optional) additional arguments for call_2
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
        sample_indices = sample_creator(distribution, m)
        data_sample = data[sample_indices]
        label_sample = labels[indices]

        new_learner = weak_learner(data_sample, label_sample)
        (epsilon, alpha, distribution) = evaluate(new_learner,
                                                    distribution,
                                                    data,
                                                    labels,
                                                    call_2,
                                                    args_2)

    return

def sample_creator(distribution, m):
    '''
    Creates a new sample of size <m> by randomly sampling from <distribution>,
    a discrete probability distribution over the cases in <data x labels>
    Inputs:
        distribution: a list of floats of length equal to that of data and labels
        m: size of the sample to be created
    Returns:
        sample_indices: a list of indices which indicate the new sample
    '''
    indices = np.arange(len(distribution))
    sample_indices = np.random.choice(indices, m, True, distribution)
    return sample_indices


def evaluate(new_learner, distribution, data, labels, call, args = None):
    '''
    Evaluates the performance of the new weak learner on the training data
    Inputs:
        weak_learner: a machine learning model produced in one iteration of
            adaboost
        distribution: the current sample distribution
        call: function call to evaluate the weak learning algorithm
    '''
    source_size = len(data)
    errors = np.zeros(source_size, dtype=int)
    if args == None:
        for i in range(source_size):
            iteration_error = call(new_learner, data[i], labels[i])
            if iteration_error:
                errors[i] = 1
    else:
        for i in range(source_size):
            iteration_error = call(new_learner, data[i], labels[i], args)
            if iteration_error:
                errors[i] = 1
    #Epsilon is the loss of the new predictor on the old distribution
    epsilon = np.sum(np.dot(errors, distribution))

    #Alpha is a measure of our confidence in the new predictor
    alpha = .5 * np.log(1 / epsilon - 1)

    #Create the nultiplicative component of the new distribution:
    #Cases with errors are up-weighted and cases without errors are down-weighted
    new_dist = np.exp(alpha * (-1)**(1 + errors))

    #Apply this to the old distribution
    new_dist = np.dot(new_dist, distribution)
    new_dist = distribution / np.sum(new_dist)

    return (epsilon, alpha, new_dist)
