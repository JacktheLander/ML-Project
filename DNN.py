# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch, os
from os.path import join, isfile
import pandas as pd

# ******************************************************************************
def DotSumAddBias(x, W, b):
	return np.add(np.dot(x, W), b)

# ******************************************************************************
def ReLU(Z):
	'''ReLU function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: ReLU values
	'''
	return np.maximum(0, Z)

# ******************************************************************************
def ReLUDerivative(Z):
	'''ReLU derivative function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: derivative of ReLU values
	'''
	Z[Z <= 0.0] = 0.0
	Z[Z > 0.0]  = 1.0
	return Z

# ******************************************************************************
def Sigmoid(Z):
	'''Sigmoid function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: Sigmoid values
	'''
	return np.divide(1.0, np.add(1.0, np.exp(-Z)))

# ******************************************************************************
def SigmoidDerivative(Z):
	'''Sigmoid derivative function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: derivative Sigmoid values
	'''
	return np.multiply(Sigmoid(Z), np.subtract(1.0, Sigmoid(Z)))

# ******************************************************************************
def Softmax(Z):
	'''Softmax function
	Inputs:
	Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: softmax values
	'''
	"""Applies softmax function row-wise."""
	ExpVals = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # For numerical stability
	return np.divide(ExpVals, np.sum(ExpVals, axis=1, keepdims=True))

# ******************************************************************************
def SoftmaxDerivative(Z): # Best implementation (VERY FAST)
	'''Softmax derivative function
		Returns the jacobian of the Softmax function for the given set of inputs.
	Inputs:
		x: should be a 2d (mxn) matrix where m corresponds to the samples
			(or mini-batch), and n is the number of nodes.
	Returns: jacobian derivative of softmax
	reference: https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-\
			activation-function-and-its-derivative-jacobian-in-python/
	'''
	s     = Softmax(Z)
	a     = np.eye(s.shape[-1])
	Temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
	Temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
	Temp1 = np.einsum('ij,jk->ijk', s, a)
	Temp2 = np.einsum('ij,ik->ijk', s, s)
	return np.average(np.subtract(Temp1, Temp2), axis=2)

# ******************************************************************************
def CrossEntropy(Outputs, Targets):
	'''Cross entropy loss function
	Inputs:
		Outputs: a 2d matrix (mxn): m mini-batch, n output neurons
		Targets: a 2d matrix (mxn): m mini-batch, n expected values
	Returns: average derivative of cross entropy loss function
	'''
	# Loss    = -np.mean(Targets * np.log(Outputs) + (1 - Targets) * np.log(1 - Outputs))
	Loss    = -np.mean(np.add(np.multiply(Targets, np.log(Outputs)), np.multiply((1.0 - Targets), np.log(1.0 - Outputs))))
	return Loss

# ******************************************************************************
def CrossEntropyDeri(Outputs, Targets):
	'''Derivative of Cross entropy loss function
	Inputs:
		Outputs: a 2d matrix (mxn): m mini-batch, n output neurons
		Targets: a 2d matrix (mxn): m mini-batch, n expected values
	Returns: average derivative of cross entropy loss function
	'''
	DeriVector = np.add(np.divide(-Targets, (Outputs * np.log(10))), np.divide((1.0 - Targets), (np.log(10) * (1 - Outputs))))
	return DeriVector

# ******************************************************************************
def LoadData(DataFolder):
	# **************************************************************************
	# display the information
	# **************************************************************************
	FunctionName = "LoadData()"

	# **************************************************************************
	# data file names
	# **************************************************************************
	TrainFile      = join(DataFolder, "TrainVectors.csv")
	TrainLblFile   = join(DataFolder, "TrainTargets.csv")
	TestFile       = join(DataFolder, "TestVectors.csv")
	TestLblFile    = join(DataFolder, "TestTargets.csv")

	# **************************************************************************
	# format the labels or targets
	# **************************************************************************
	TrainVectors   = np.loadtxt(TrainFile, delimiter=",", skiprows=1)
	TrainTargets   = np.loadtxt(TrainLblFile, delimiter=",", skiprows=1)
	TestData       = np.loadtxt(TestFile, delimiter=",", skiprows=1)
	TestTargets    = np.loadtxt(TestLblFile, delimiter=",", skiprows=1)
	return TrainVectors, TrainTargets, TestData, TestTargets

# ******************************************************************************
"""
 Feedforward Neural Network: http://neuralnetworksanddeeplearning.com/chap2.html
 Lab 3: Implementing Backpropagation Gradien Descent algorithm on a feedforward
 nework
"""
class BPFFNetwork:
	def __init__(self, InParams, Verbose=False):
		""" Initializes and constructs a feedforward network.
		Parameters are:
			- NetStructure  : a list of neurons [input, hidden, ..., outputs].
							  For example, [7, 3, 4, 3] means a FF network of 7x3x4x3
			- Epoch			: a number of trainings
			- Batch         : mimi-batch number. The default value should be 1.
			- Eta           : learning rate. The default value should be 0.001.
		"""
		# **********************************************************************
		# network name
		# **********************************************************************
		self.NetworkName  = "BPFFNetwork"

		# **********************************************************************
		# save the initial parameters
		# **********************************************************************
		self.NetStruct    = InParams["NetStruct"]
		self.Epoch        = InParams["Epoch"]
		self.Batch        = InParams["Batch"]
		self.LearnRateEta = InParams["LearnRateEta"]
		self.Verbose      = Verbose

		# Initialize Weights and Biases randomly - size determined by number of neurons in NetStruct

		self.W = [
			np.random.randn(self.NetStruct[0], self.NetStruct[1]),  # Weights for Layer 1
			np.random.randn(self.NetStruct[1], self.NetStruct[2]),  # Weights for Layer 2
			np.random.randn(self.NetStruct[2], self.NetStruct[3])  # Weights for Layer 3
		]

		self.B = [
			np.random.randn(1, self.NetStruct[1]),  # Biases for Layer 1
			np.random.randn(1, self.NetStruct[2]),  # Biases for Layer 2
			np.random.randn(1, self.NetStruct[3])  # Biases for Layer 3
		]
		#...

	# **************************************************************************
	'''
		This function calculates a Forward pass of the network.
		Inputs: x[b, m] -> b: batch number, m: input neurons
		Output: o[b, n] -> b: batch number, n: output neurons
	'''
	def _Forward(self, x):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName    = "::_Forward():"

		self.a0 = x
		# Layer 1 Input to Hidden1 using ReLU
		self.z1 = DotSumAddBias(x, self.W[0], self.B[0])
		self.a1 = ReLU(self.z1)

		# Layer 2 Hidden1 to Hidden2 using ReLU
		self.z2 = DotSumAddBias(self.a1, self.W[1], self.B[1])
		self.a2 = ReLU(self.z2)

		# Layer 3 Hidden2 to Output using Sigmoid
		self.z3 = DotSumAddBias(self.a2, self.W[2], self.B[2])
		self.a3 = Sigmoid(self.z3)

		return self.a3
		#...

	# **************************************************************************
	'''
		This function calculates backpropagate errors:
			NablaW = nabla weights
			NablaB = nabla bias
	'''
	def _BackProp(self, Outputs, TrainTargets):

		# Error Calculations start with cost and pass back to each layer
		# cost = CrossEntropyDeri(Outputs, TrainTargets) # Previously I used cost to calculate output error with softmax derivative
		output_error = np.asarray(Outputs).reshape(-1) - np.asarray(TrainTargets).reshape(-1) 	# Softmax derivative is not necessary it can be simplified to this eq
		output_error = output_error.reshape(-1, 1)

		h2_error = np.multiply(ReLUDerivative(self.z2), np.dot(output_error, self.W[2].T)) #  Eq.s from module 3 pg. 55
		h1_error = np.multiply(ReLUDerivative(self.z1), np.dot(h2_error, self.W[1].T))

		# Calculate Nablas
		NablaBiases = [h1_error, h2_error, output_error]
		NablaWeights = []

		#print(f"h1_error shape: {h1_error.shape}, a1 shape: {self.a1.shape}, W[0] shape: {self.W[0].shape}")
		#print(f"h2_error shape: {h2_error.shape}, a2 shape: {self.a2.shape}, W[1] shape: {self.W[1].shape}")
		#print(f"output_error shape: {output_error.shape}, a3 shape: {self.a3.shape}, W[2] shape: {self.W[2].shape}")

		NablaWeights.append(np.dot(self.a0.T, h1_error))
		NablaWeights.append(np.dot(self.a1.T, h2_error))
		NablaWeights.append(np.dot(self.a2.T, output_error))


		#print(f"NablaWeights[0] shape: {NablaWeights[0].shape}, expected: (10, 15)")
		#print(f"NablaWeights[1] shape: {NablaWeights[1].shape}, expected: (15, 22)")
		#print(f"NablaWeights[2] shape: {NablaWeights[2].shape}, expected: (22, 4)")

		self._UpdateLayerWeights(NablaWeights, NablaBiases)
		return
		#...

	# **************************************************************************
	def _UpdateLayerWeights(self, NablaWeights, NablaBiases):
		# Update the weights by a factor of eta

		for i in range(len(self.W)):
			#print(f"Before update: Layer {i} - Weight Norm: {np.linalg.norm(self.W[i])}")
			#print(NablaWeights[i])
			self.W[i] -= self.LearnRateEta * NablaWeights[i]
			self.B[i] -= self.LearnRateEta * np.sum(NablaBiases[i], axis=0, keepdims=True)
			#print(f"After update: Layer {i} - Weight Norm: {np.linalg.norm(self.W[i])}")
		return
		#...

	# **************************************************************************
	'''
		This function trains the network
		Inputs:
			TrainData[l, m]		-> l: the number of input data, m: input neurons
			TrainTargets[l, n]	-> l: the number of input data, n: output neurons	
	'''
	def TrainNetwork(self, TrainData, TrainTargets):

		TrainData = pd.DataFrame(TrainData).reset_index(drop=True)
		TrainTargets = pd.Series(TrainTargets).reset_index(drop=True)

		num_samples = TrainData.shape[0]  # Number of training samples
		num_batches = num_samples // self.Batch  # Number of batches per epoch

		for epoch in range(self.Epoch):
			# Shuffle data for each epoch to improve generalization
			indices = np.arange(num_samples)
			#np.random.shuffle(indices)
			TrainData, TrainTargets = TrainData.iloc[indices], TrainTargets.iloc[indices]

			epoch_loss = 0  # To store total loss in this epoch

			for i in range(num_batches):
				# Extract mini-batch
				start_idx = i * self.Batch
				end_idx = start_idx + self.Batch
				batch_data = TrainData[start_idx:end_idx]
				batch_targets = TrainTargets[start_idx:end_idx]

				# Forward pass
				outputs = self._Forward(batch_data)
				outputs = outputs.reshape(-1)	# Mismatched output size fix, flatten array to 1D

				# Compute loss
				#print("outputs:", type(outputs), outputs.shape)
				#print("targets:", type(batch_targets), batch_targets.shape) Debugging Mismatched output size
				loss = CrossEntropy(outputs, batch_targets)
				epoch_loss += loss

				# Backpropagation & weight update
				self._BackProp(outputs, batch_targets)
				#print(self.W)

			# Logging (Verbose Mode)
			if self.Verbose:
				avg_loss = epoch_loss / num_batches
				print(f"Epoch [{epoch + 1}/{self.Epoch}], Loss: {avg_loss:.6f}")

		print("Training Completed!")
		return
		#...

	# **************************************************************************
	def _CalPerf(self, Outputs, TestLbls):
		# 1) Ensure numpy arrays of same shape
		preds = (np.asarray(Outputs).reshape(-1) >= 0.5).astype(int)
		true = np.asarray(TestLbls).reshape(-1).astype(int)

		# 2) Compute accuracy
		accuracy = np.mean(preds == true) * 100.0
		return accuracy

	# **************************************************************************
	'''
		This function test the network
		Inputs:
			TestData[k, l]		-> k: the number of input data, l: input neurons
			TestTargets[k, n]	-> k: the number of input data, m: output neurons
	'''
	def TestNetwork(self, TestData, TestTargets):
		# 1) coerce targets to a 1-D numpy array
		y_true = np.asarray(TestTargets).reshape(-1)

		# 2) forward pass and flatten predictions
		outputs = self._Forward(np.asarray(TestData)).reshape(-1)

		# 3) compute loss (no extra division)
		loss = CrossEntropy(outputs, y_true)

		# 4) compute accuracy
		accuracy = self._CalPerf(outputs, y_true)

		if self.Verbose:
			print(f"Test Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")

		return loss, accuracy
	#...

# ******************************************************************************
if __name__ == "__main__":
	# **************************************************************************
	# load data to np array
	# **************************************************************************
	TrainData, TrainTargets, TestData, TestTargets = LoadData("Assets")
	# print("TrainData = ", TrainData.shape)
	# print("TrainTargets = ", TrainTargets.shape)
	# **************************************************************************
	# parameters for the network
	# **************************************************************************
	NetStruct = [10, 15, 22, 4]
	Epoch = 50
	Batch = 8
	LearnRateEta = 0.05
	Bias = [0.1, 0.2, 0.3, 0.4]
	Verbose = True
	InParams = { "NetStruct": NetStruct, "Epoch": Epoch, "Batch": Batch, "LearnRateEta": LearnRateEta, "Bias": Bias}
	# **************************************************************************
	# create a feedforward network
	# **************************************************************************
	NetFF = BPFFNetwork(InParams, Verbose=Verbose)
	# **************************************************************************
	# train network
	# **************************************************************************
	NetFF.TrainNetwork(TrainData, TrainTargets)
	# **************************************************************************
	# test network
	# **************************************************************************
	Performance = NetFF.TestNetwork(TestData, TestTargets)
