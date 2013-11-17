clear all;
close all;
clc;

%% load data
load('data1.mat');
%load('data2.mat')

K = max(labels); % number of clusters

%% display data
fx = figure;
set(fx, "visible", "off");
scatter(X(1,:),X(2,:),'b');
axis equal;
title('data');

colors = {'b', 'r', 'g', 'y'};

f0 = figure;
set(f0, "visible", "off");

for j = 1:K
    scatter( X(1,labels == j), X(2,labels == j), colors{j} );
    hold on;
end
axis equal;
title('data with ground truth cluster labelling');
print(f0, "ground_truth.png", "-dpng");

%% cluster with k-means
perm = randperm(size(X,2));
Cinit = X(:,perm(1:K));

%Cinit = [0, 0; 1, 1; 2, 2]';
%Cinit = [0, 0; 8, 2; 10, 5]';

fprintf('k-means ... \n');
tic;
[C, A] = Kmeans_PASCALZAUGG(X, Cinit);
toc;

%% display clustering
f1 = figure;
set(f1, "visible", "off");

for j = 1:K
    scatter( X(1,A == j), X(2,A == j), colors{j});
    hold on;
    scatter( C(1,j), C(2,j), 100, colors{j}, 'x');
    scatter( Cinit(1, j), Cinit(2, j), 20, 'black', 'x');
end
axis equal;
title('clustered data');
print(f1, "clustered.png", "-dpng");
