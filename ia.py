class IA:
    
    def __init__(self,layers:list) -> None:
        self.__layers = layers
        self.__neurons = []
        self.__weights = []
        self.__init_neurons()
        self.__init_weights()

    def __init_weights(self,):
        [[self.__weights.append(2*np.random.random((len(self.__neurons[idx]), len(self.__neurons[idx+1])))-1)] for idx in range(0, len(self.__neurons)-1)]

    def __init_neurons(self,):
        [self.__neurons.append([0 for i in range(0, self.__layers[idx])]) for idx in range(0, len(self.__layers))]
    
    def __feed_fordward(self,inputs):
        self.__neurons[0] = inputs
        for idx in range(1,len(self.__neurons)):
            self.__neurons[idx] = tools.sigmoid(np.dot(self.__neurons[idx-1], self.__weights[idx-1]))
    
    def __back_propagation(self,responses):
        error = (responses - self.__neurons[len(self.__neurons)-1])
        delta = []        
        delta.append(error*tools.sigmoidPrime(self.__neurons[len(self.__neurons)-1]))
        for i in reversed(range(0, len(self.__neurons)-1)):
            hidden_layer_error = np.dot(delta[len(self.__neurons)-2-i], self.__weights[i].T)
            delta.append(hidden_layer_error*tools.sigmoidPrime(self.__neurons[i]))
        for i in reversed(range(0, len(self.__weights))):
            self.__weights[i] += np.dot(self.__neurons[i].T, delta[len(self.__weights)-1-i])
