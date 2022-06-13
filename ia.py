class IA:
    
    def __init__(self,name:str,layers:list):
        self.__name = name
        self.__layers = layers
        self.__neurons = []
        self.__weights = []
        self.__init_network()

    def __init_network(self,) -> bool:
        self.__init_neurons():
        self.__init_weights()

    def __init_neurons(self,) -> bool:
        [self.__neurons.append(np.array([0 for i in range(0, self.__layers[idx])])) for idx in range(0, len(self.__layers))]

    def __init_weights(self,) -> bool:
        [[self.__weights.append(2*np.random.random((len(self.__neurons[idx]), len(self.__neurons[idx+1])))-1)] for idx in range(0, len(self.__neurons)-1)]

    def propagation(self,inputs) -> bool:
        self.__neurons[0] = inputs
        for idx in range(1,len(self.__neurons)):
            self.__neurons[idx] = tools.sigmoid(np.dot(self.__neurons[idx-1], self.__weights[idx-1]))
        self.__back_propagation_is_ready = True

    def back_propagation(self,responses) -> bool:
        error = (responses - np.array(self.__neurons[len(self.__neurons)-1]))
        delta = []        
        delta.append(error*tools.sigmoidPrime(self.__neurons[len(self.__neurons)-1]))
        self.__price =  str(np.mean(np.abs(error)))
        for i in reversed(range(1, len(self.__weights))):
            error = np.dot(delta[len(self.__weights)-1-i], self.__weights[i].T)
            delta.append(error*tools.sigmoidPrime(self.__neurons[i]))
        for i in reversed(range(0, len(self.__weights))):
            self.__weights[i] += np.dot(self.__neurons[i].T, delta[len(self.__weights)-1-i])
        self.__back_propagation_is_ready = False