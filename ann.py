
import numpy as np





def sgm(x,derivative=False):
	if not derivative:
		return 1/(1+np.exp(-x))
	else:
		out = sgm(x)
		return out*(1.0-out)

def linear(x,derivative=False):
	if not derivative:
		return x
	else:
		return 1.0

def guassian(x,derivative=False):
	if not derivative:
		return np.exp(-x**2)
	else:
		return -2*x*np.exp(-x**2)

def tanh(x,derivative=False):
	if not derivative:
		return np.tanh(x)
	else:
		return (1.0-np.tanh(x)**2)


class backPropagationNetwork:
	layerCount = 0
	shape = None
	weights = []
	layerTransferFunc = []

	# constructor function for the class
	def __init__(self, layerSize,layerTransferFunc = None):
		# layerSize is the Architecture of the NN as (4,3,1)
		self.layerCount = len(layerSize)-1 # input layers is just a buffer
		self.shape = layerSize

		# for the forward pass maybe.
		self._layerInput = []
		self._layerOutput = []
		self._previousWeightDelta = []

		
		for(l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
			# add for each weight matrix a matrix in _previousWeightDelta for previous values
			self._previousWeightDelta.append(np.zeros(shape=(l2,l1+1)))

		if layerTransferFunc is None:
			layerTransferFunc = []
			for i in range(self.layerCount):
				if i == self.layerCount - 1:
					layerTransferFunc.append(linear)
				else:
					layerTransferFunc.append(sgm)
		else:
			if len(layerTransferFunc) != len(layerSize):
				raise ValueError("Incompatible no of transfer functions.")
			elif layerTransferFunc[0] is not None:
				raise ValueError("no transfer functions for input layer.")
			else:
				layerTransferFunc = layerTransferFunc[1:]

		self.layerTransferFunc = layerTransferFunc


	# forward run/pass
	def run(self,X):
		
		# no of training examples
		m = X.shape[0]

		# initialize/ clear out the input and output list from previous run
		self._layerInput = []
		self._layerOutput = []

		# Forward pass
		for i in range(self.layerCount):
			if i == 0:
				layerInput = self.weights[0].dot(np.vstack([X.T, np.ones([1,m])]))            # vstack(a,b) stacks matrix/vector b below matrix/vector a
			else:
				layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1,m])]))
			
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.layerTransferFunc[i](layerInput))

		return self._layerOutput[-1].T


	def trainEpoch(self,X,Y,trainingRate ,momentum ):
		# trains the network for one epoch
		delta = []
		m = X.shape[0]

		# forward pass before we can compute the gradient by back propagation
		self.run(X)

		
		for i in reversed(range(self.layerCount)):                                             # reverse as the backpropogation work in reverse order
			
			if i == self.layerCount-1:                                                         # if this is for the preactivation at the output								
				outputDelta = self._layerOutput[i] - Y.T                                      # this is also the gradient at output if we take the least square error function
				error = np.sum(outputDelta**2)                                                # sum of all the elements along all dimensions
				delta.append(outputDelta*self.layerTransferFunc[i](self._layerInput[i],True))                  # '*' operator is for coordinate wise multiplication
			else:
				deltaPullback = self.weights[i+1].T.dot(delta[-1]) 							  # this is the gradient at the activation of the hidden layer (i+1), note that i = 0
																							  # is for hidden layer 1.
				delta.append(deltaPullback[:-1,:]*self.layerTransferFunc[i](self._layerInput[i],True))         # this is the gradient at the preactivation at hidden layer (i+1)
				
		for i in range(self.layerCount):
			deltaIndex = self.layerCount - 1 - i 												# delta[0] is preactivation at output and so on in backward direction
			if i == 0:
				layerOutput = np.vstack([X.T,np.ones([1,m])])									# for W0 the delta (preactivation) is input layers
			else:
				layerOutput = np.vstack([self._layerOutput[i-1],np.ones([1,self._layerOutput[i-1].shape[1]])])		# _layerOutput[0] contains the activation of the hidden layer 1 and so for Wi we need _layerOutput[i-1] 

			weightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1)*delta[deltaIndex][None,:,:].transpose(2,1,0),axis=0)
		
			weightDelta = trainingRate * weightDelta + momentum*self._previousWeightDelta[i]
			self.weights[i] -= weightDelta
			self._previousWeightDelta[i] = weightDelta
			
		return error # incase useful




if __name__ == "__main__":
    
    print ("welcome to neural network")
    data_file=raw_input("please mention the name of your training set file\n")
    
    d=np.genfromtxt(str(data_file),delimiter=',')
    z=np.shape(d)
    print ("your data set has %i datapoints a\n"%z[0]) 
    print ("And there are %i columns\n"%z[1])
    nx=int(raw_input("how many of the %i columns are inputs ?\n"%z[1]))
    ny=z[1]-nx
    x=[]
    y=[]
    for p in range(z[0]):
        x.append(d[p,0:nx])
        y.append(d[p,nx:])
    X = np.array(x)
    Y= np.array(y)
    net_def=[int(nx)]
    print net_def
    qwe=raw_input("how many epochs ?\n")
    hlc=int(raw_input("how many hidden layer do you want ?\n"))
    print int(hlc)
    for n in range(int(hlc)):
        a=raw_input("how many units in layer %s\n"%(n+2))
        net_def.append(int(a))
    net_def.append(ny)
    m=float(raw_input("please define momentum:\n"))
    alph=float(raw_input("please define learning rate:\n"))
    print("You can choose from a host of transfer functions :-sigmoid(sgm),linear(linear),gaussian(gaussian),tanh(tanh)\n")
    tff=str(raw_input("choose on of the transfer functions(mentioned in paranthesis above)\n"))
    tup=tuple(net_def)
    layerTransferFunc = [None]
    for n in range(len(tup)-1):
        layerTransferFunc.append(str(tff))
    bpn = backPropagationNetwork(tup)
    maxIteration = int(qwe)
    minError = 1e-5
   
    for i in range(maxIteration+1):
        
        
        err = bpn.trainEpoch(X,Y,momentum=float(m),trainingRate=float(alph))
       
        if i%100 == 0:
            print "iteration {0}\t error: {1:0.6f}".format(i, err)
        if err <= minError:
            print "Minimum error reached as iteration {0}".format(i)
            break
    Ycomputed = bpn.run(X) 
    for i in range(np.shape(X)[0]):
        print "Input: {0}\n Output: {1}".format(X[i],Ycomputed[i])#tup
    print(bpn.shape)
    print(bpn.weights)
    
	

	
	

	
	

	
	
	
		
		
			
		
			
			

	

	
		
	
