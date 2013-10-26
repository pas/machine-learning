function [phi, mu0, mu1, Sigma] = GDATrain( DataTrain, LabelsTrain )

LabelsTrain = convert(LabelsTrain);
dim = size(DataTrain,2);

testFunctions();

phi = calculatePhi(LabelsTrain);
mu0 = calculateMu0(DataTrain, LabelsTrain, dim);
mu1 = calculateMu1(DataTrain, LabelsTrain, dim);
Sigma = calculateSigma(DataTrain, LabelsTrain, dim, mu0, mu1);

end

function DataTrain = convert(DataTrain) 
	DataTrain(DataTrain == -1) = 0;
endfunction

function phi = calculatePhi( LabelsTrain )
  m = size(LabelsTrain, 2);

  %sum all entries in LabelsTrain with one
  result = sum(LabelsTrain(:)==1);
  phi = result/m;
endfunction

function mu0 = calculateMu0(DataTrain, LabelsTrain, dimension)
  %count labels with 0
  sumOfZeros = sum(LabelsTrain(:)==0, 1);
  %sum all rows which correspond to the label 0
  result = sum(DataTrain(LabelsTrain==0, :), 1);
  %divide result by the count of labels
  mu0 = transpose(result / sumOfZeros);
endfunction

function mu1 = calculateMu1(DataTrain, LabelsTrain, dimension)
  %count labels with 1
  sumOfOnes = sum(LabelsTrain(:)==1);
  %sum all rows which correspond to the label 1
  result = sum(DataTrain(find(LabelsTrain), :), 1);
  %divide result by the count of labels
  mu1 = transpose(result / sumOfOnes);
endfunction

function sigma = calculateSigma(DataTrain, LabelsTrain, dimension, mu0, mu1)
  m = size(LabelsTrain, 2);

  positive = DataTrain(LabelsTrain==1, :);
  positiveSize = size(positive, 1);
  negative = DataTrain(LabelsTrain==0, :);
  negativeSize = size(negative, 1);

  mu0Matrix = repmat(mu0', negativeSize, 1);
  negative = bsxfun(@minus, negative, mu0');

  mu1Matrix = repmat(mu1', positiveSize, 1);
  positive = bsxfun(@minus, positive, mu1');

  sigma = ((positive' * positive)+(negative' * negative))/m;
endfunction

function testFunctions()
  testData = [0, 17; 1, 7; 4, 3; 1, 2]; 
  testLabels = [0, 0, 1, 1];
  sum(testData(testLabels==0, :))

  % testing calculation of phi
  testLabels = [0, 1, 1, 1];
  phi = calculatePhi(testLabels);
  assert(phi == 0.75);

  % testing calculation of mu0
  mu0 = calculateMu0(testData, testLabels, 2);
  assert(mu0 == [0; 17]);

  % testing calculation of mu1
  mu1 = calculateMu1(testData, testLabels, 2);
  assert(mu1 == [2; 4]);

  % testing calculation of sigma
  sigma = calculateSigma(testData, testLabels, 2, mu0, mu1);
  assert(sigma == [1.5, -0.75; -0.75, 3.5]);
endfunction

