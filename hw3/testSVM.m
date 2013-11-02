close all;
clear all;
clc;

%%
% Training 
fprintf('Training ... ');
VLFEAT_FOLDER = '../vlfeat'; %Put here the path to the root floder of VlFeat

addpath ([VLFEAT_FOLDER, '/toolbox']);
vl_setup;

X = [1, 2; 2, 1; 5, 7; 7, 5; 3, 3; 4, 4; 1, 1]
y = [1; 1; -1; -1; -1; -1; -1];

X = X';
y = y';

Xp = X(:, y==1);
Xn = X(:, y==-1);

f = figure;
set(f, "visible", "off");

plot(Xn(1,:),Xn(2,:),'*r')
hold on
plot(Xp(1,:),Xp(2,:),'*b')
axis equal ;

lambda = 0.02 ; % Regularization parameter
maxIter = 1000 ; % Maximum number of iterations

[w b info] = vl_svmtrain(X, y, lambda, 'MaxNumIterations', maxIter);

result = w' * X + b;
result(result > 0) = 1;
result(result <= 0) = -1;
positive = sum(result == 1)
negative = sum(result == -1)

% Visualisation
eq = [num2str(w(1)) '*x+' num2str(w(2)) '*y+' num2str(b)];
line = ezplot(eq);
set(line, 'Color', [0 0.8 0],'linewidth', 2);

print(f, "testResultPlot.png", "-dpng");
