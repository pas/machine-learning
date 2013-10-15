close all;
clear all;
clc;

%%
% Data contains our training examples. It is an Nx576 matrix. Each row
% represents one train example. The training examples are 24x24 grayscale
% images represented as a 1x576 row vector. The range of possible values
% are from 0 to 255. The dataset contains the mirrored pair of every face
% in the dataset.

% Labels contain the labesl of the training data. It is an 1xN matrix. If
% Labels(1,k) = 1, it means the kth sample is a face, if it is -1, than the
% sample is not a face. All the labels are either 1 or -1.

load('faces.mat');
N = size(Data,1);

%%
% Here we plot some examples images from the database. Notice, that the
% dataset contains the mirrores images.

faceIdx = Labels > 0;
nonFaceIdx = Labels < 0;
figure;
n = 6; % number of examples in a row
% faces (first row)
FaceData = Data(faceIdx,:);
for k = 1:n
    im = reshape(FaceData(k,:), [24 24]);
    subplot(2,n,k);
    imshow( im/255 );
    if k==1
        title( 'Faces' );
    end
end
% non faces (second row)
NonFaceData = Data(nonFaceIdx,:);
for k = 1:n
    im = reshape(NonFaceData(k,:), [24 24]);
    subplot(2,n,k+n);
    imshow( im/255 );
    if k==1
        title( 'Non faces' );
    end
end

%%
% First we select the train and test data randomly. We select aproximately
% testPercent% images for test images from the data. testIdx(1,k) = 1 if
% Data(k,:) belongs to the test set and 0 if it does not. trainIdx is
% defined similarly.
% We also add the intercept term to the data.

testPercent = 30; 
testIdx = rand(1,N) <= testPercent/100;
trainIdx = ~testIdx;

DataI = [ ones(N,1), Data ]; % add intercept term

DataTrainI = DataI(trainIdx,:);
LabelsTrain = Labels(1,trainIdx);
DataITestI = DataI(testIdx,:);
LabelsTest = Labels(1,testIdx);

%%
% Training with Logistic Regression.
fprintf('Training ... ');
tic;
theta = logisticRegressionTrain( DataTrainI, LabelsTrain , 5);
% theta = logisticRegressionTrain_SOLUTION( DataTrainI, LabelsTrain , 5);
toc;

%%
% Test logistic regression
scores = 1 ./ (1 + exp(-DataITestI*theta)');
classifierOutput = (scores >= 0.5) - (scores < 0.5);

%%
% Evaluation.
% The metrics of evaluations (ROC and precision-recall curve) are described
% on Wikipedia:
%     http://en.wikipedia.org/wiki/Precision_and_recall
%     http://en.wikipedia.org/wiki/Receiver_operating_characteristic

Ntest = size(DataITestI,1);
[sortedScores,idx] = sort(scores,'descend'); 
sortedLabelsTest = LabelsTest(idx);
positives = sortedLabelsTest == 1;

% overall accuracy
good = classifierOutput == LabelsTest;
fprintf( 'accuracy: %f%% (%d/%d)\n', 100*sum(good)/size(good,2), sum(good), size(good,2) );

% precison - recall curve
precision = cumsum( positives ) ./ (1:Ntest);
recall = cumsum( positives ) / sum( positives );
figure;
plot(recall,precision);
axis( [0 1 0 1] );
title('precision-recall curve');
xlabel('Recall');
ylabel('Precision');

% ROC curve
truePosRate = cumsum( positives ) / sum( positives ); % same as recall
falseNegRate = cumsum( ~positives ) / sum( ~positives );
figure;
plot(falseNegRate,truePosRate);
axis( [0 1 0 1] );
title('ROC curve');
xlabel('False Negative Rate');
ylabel('True Positive Rate');


% plot some classifications
figure;
n = 10; % number of examples in a row
m = 3; % number of examples in a column
DataTest = Data(testIdx,:);
% faces (first row)
ClassifiedFace = DataTest( classifierOutput>0 ,:);
FaceGood = good( classifierOutput>0 );
perm1 = randperm( sum(classifierOutput>0) );
for k = 1:min(3*n,sum(classifierOutput>0))
    im = reshape( ClassifiedFace(perm1(k),:), [24 24] );
    
    color = FaceGood(perm1(k))+1; % red = 1 and green = 2 
    imrgb = zeros(26,26,3);
    imrgb(:,:,color) = 255;
    imrgb(2:25,2:25,1) = im;
    imrgb(2:25,2:25,2) = im;
    imrgb(2:25,2:25,3) = im;
    
    subplot(2*m,n,k);
    imshow( imrgb/255 );
    if k==1
        title( 'Classified as face' );
    end
end
% non faces (second row)
ClassifiedAsNonFace = DataTest( classifierOutput<0 ,:);
NonFaceGood = good( classifierOutput<0 );
perm2 = randperm( sum(classifierOutput<0) );
for k = 1:min(3*n,sum(classifierOutput<0))
    im = reshape( ClassifiedAsNonFace(perm2(k),:), [24 24] );
    
    color = NonFaceGood(perm2(k))+1; % red = 1 and green = 2
    imrgb = zeros(26,26,3);
    imrgb(:,:,color) = 255;
    imrgb(2:25,2:25,1) = im;
    imrgb(2:25,2:25,2) = im;
    imrgb(2:25,2:25,3) = im;
    
    subplot(2*m,n,k+m*n);
    imshow( imrgb/255 );
    if k==1
        title( 'Classified as non-face' );
    end
end





