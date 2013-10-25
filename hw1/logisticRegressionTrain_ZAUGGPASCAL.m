function [ theta ] = logisticRegressionTrain_ZAUGGPASCAL( DataTrain, LabelsTrain, maxIterations )
% logisticRegressionTrain train a logistic regression classifier
% [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, MaxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 
% 
% Implement a Newton-Raphson algorithm.

	testFunctions();
	theta = runAlgorithm(DataTrain, LabelsTrain, maxIterations);

end

function theta = runAlgorithm(DataTrain, LabelsTrain, maxIterations)
	% number of training samples
	dim = size(DataTrain,2);
	m = size(DataTrain,1);
	assert(m == size(LabelsTrain, 2));

	%convert labels of training data to fit 1=1 and -1=0
	LabelsTrain = convert(LabelsTrain);

	% define theta-vector and fill with zeros
	theta = zeros(dim,1);

	for iterationNr = 1:maxIterations
		hessianInverted = inverse(hessianLogL(dim, m, theta, DataTrain, LabelsTrain));
		gradientVector = gradientLogL(dim, m, theta, DataTrain, LabelsTrain);

		theta = theta - hessianInverted * gradientVector;
	end
endfunction

function DataTrain = convert(DataTrain) 
	DataTrain(DataTrain == -1) = 0;
endfunction

function result = hessianLogL(dimension, sampleSize, theta, DataTrain, LabelsTrain)

	%define matrix with zeros
	result = zeros(dimension);

	%calculate hessian by iterating over
	%all samples

	for i = 1:sampleSize
		%Transpose because data is organised in rows
		x = transpose(DataTrain(i, :));
	
		result += h(theta, x) * (1-h(theta, x)) * x * transpose(x);
	end

	result = -result/sampleSize;

endfunction

function result = sigmoid(x) 

	result = 1 / (1 + exp(-x));

endfunction

function result = h(theta, x)

	result = sigmoid(transpose(theta)*x);

endfunction

function result = gradientLogL(dimension, sampleSize, theta, DataTrain, LabelsTrain)

	%define vector with zeros
	result = zeros(dimension, 1);

	%calculate gradient by summing everything up
	%and then multiply the result with 1/sampleSize

	%simulates sum function
	for i = 1:sampleSize
		%Getting label for current sample
	  	y = LabelsTrain(i);
		%Getting data for current sample
		%Transpose because data is organised in rows
		x = transpose(DataTrain(i, :));

		%adding results up
		result += (y - h(theta, x))*x;
	end

	result = 1/sampleSize * result;

endfunction

function testFunctions
	%Test sigmoid(x)
	resultOfG = sigmoid(0);
	assert(resultOfG == 1/2);

	%Test h
	testX = [0; 0];
	testTheta = [1; 0];
	resultOfHTheta = h(testTheta, testX);
	assert(resultOfHTheta == 1/2);

	%Declare variables for hessian and gradient test
	testTheta = [0; 0];
	testDataSet = [0, 1; 0, 0; 1, 0; 1, 1];
	testDataSize = size(testDataSet,1);
	testDataDimension = size(testDataSet, 2);
	testLabelSet = [1; -1; 1; 1];

	%Test data conversion
	convertedLabelSet = convert(testLabelSet);
	assert(convertedLabelSet == [1; 0; 1; 1]);

	%Test gradient
	gradientVector = gradientLogL(testDataDimension, testDataSize, testTheta, testDataSet, convertedLabelSet);
	assert(gradientVector == [1/4; 1/4]);

	%Test hessian
	hessian = hessianLogL(testDataDimension, testDataSize, testTheta, testDataSet, convertedLabelSet);
	%Is this even a correct result?
	assert(hessian == [-0.125, -0.0625; -0.0625, -0.125]);
endfunction
