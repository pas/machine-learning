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
  m = size(LabelsTrain, 2);

  %sum all entries in LabelsTrain with zero
  sumOfZeros = sum(LabelsTrain(:)==0);
  
  result = zeros(dimension, 1);

  for index = 1:m 
    x = transpose(DataTrain(index, :));
    y = LabelsTrain(index);
    
    if(y == 0) 
      result = result + x;
    end
  end

  mu0 = result / sumOfZeros;
endfunction

function mu1 = calculateMu1(DataTrain, LabelsTrain, dimension)
  m = size(LabelsTrain, 2);

  sumOfOnes = sum(LabelsTrain(:)==1);

  result = zeros(dimension, 1);
  control = 0;

  for index = 1:m 
    x = transpose(DataTrain(index, :));
    y = LabelsTrain(index);

    if(y == 1)
       result = result + x;
    endif
  endfor

   mu1 = result / sumOfOnes;
endfunction

function sigma = calculateSigma(DataTrain, LabelsTrain, dimension, mu0, mu1)
  m = size(LabelsTrain, 2);
  sigma = zeros(dimension);

  for index = 1:m
    x = transpose(DataTrain(index, :));
    y = LabelsTrain(index);

    mu = mu1;
    if(y == 0)
      mu = mu0;
    end

    sigma = sigma + ((x - mu)*transpose(x - mu));        
  end

  sigma = sigma/m;
endfunction

function testFunctions()
  % testing calculation of phi
  testLabels = [0, 1, 1, 1];
  phi = calculatePhi(testLabels);
  assert(phi == 0.75);

  % testing calculation of mu0
  testData = [0, 1; 1, 7; 4, 3; 1, 2]; 
  mu0 = calculateMu0(testData, testLabels, 2)
  assert(mu0 == [0; 1]);

  % testing calculation of mu1
  mu1 = calculateMu1(testData, testLabels, 2)
  assert(mu1 == [2; 4]);

  % testing calculation of sigma
  sigma = calculateSigma(testData, testLabels, 2, mu0, mu1)
  %assert(sigma == [0.25, 0; 0, 0]);
endfunction

