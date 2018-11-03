def adaboost(weak_learner, m, T, data):
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
    '''

    #Initialization
    source_size = len(data)
    initial_distribution = [1/source_size]*source_size
    return(initial_distribution)
