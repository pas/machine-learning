function [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, maxIterations )
% logisticRegressionTrain train a logistic regression classifier
% [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, MaxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 
% 
% Implement a Newton-Raphson algorithm.


% number of training samples
m = size(DataTrain,2);
printf("%d", m);

dim = size(DataTrain, 2);
theta = zeros(dim,1);
theta(250) = -1/dim;
theta(300) = 1/dim;

%Test htheta
testTheta = [0; 1; 0];
testX = [0; 2; 0];
resultOfHTheta = htheta(testTheta, testX)
assert(resultOfHTheta == 2);

%Test g(x)
resultOfG = g(0);
assert(resultOfG == 1/2);

end

function result = g(x) 

result = 1 / (1 + exp(-x));

endfunction

function result = htheta(theta, x)

result = transpose(theta)*x;

endfunction