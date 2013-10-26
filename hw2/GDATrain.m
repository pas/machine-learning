function [phi, mu0, mu1, Sigma] = GDATrain( DataTrain, LabelsTrain )

  %convert labels from -1 to 0
  LabelsTrain(LabelsTrain == -1) = 0;

  %getting dimension and m
  dim = size(DataTrain,2);
  m = size(LabelsTrain, 2);

  %create index vector for 1
  onesIndexVector = LabelsTrain(:)==1;
  %create index vector for 0
  zerosIndexVector = LabelsTrain(:)==0;

  %count labels with 0
  sumOfZeros = sum(zerosIndexVector, 1);
  %count labels with 1
  sumOfOnes = sum(onesIndexVector, 1);

  %create matrix only holding data for 1
  positive = DataTrain(onesIndexVector, :);
  %create matrix only holding data for 0
  negative = DataTrain(zerosIndexVector, :);

  %
  % --- calculate phi ---
  %

  result = sum(onesIndexVector);
  phi = result/m;

  %
  % --- calculate mu0 ---
  % 

  %sum all rows which correspond to the label 0
  result = sum(negative, 1);
  %divide result by the count of labels
  mu0 = transpose(result / sumOfZeros);

  %
  % --- calculate mu1 ----
  %

  %sum all rows which correspond to the label 1
  result = sum(positive, 1);
  %divide result by the count of labels
  mu1 = transpose(result / sumOfOnes);
  
  %
  % --- calculate sigma ---
  %

  %subtract mu0 from each row
  negative = bsxfun(@minus, negative, mu0');

  %subtract mu1 from each row
  positive = bsxfun(@minus, positive, mu1');

  %part of sigma is sum x(i)-mu * x(i)'-mu which is exaclty the same as A'*A when
  %mu is already subtracted. 
  Sigma = ((positive' * positive)+(negative' * negative))/m;
end
