function [C, A] = Kmeans_PASCALZAUGG(X, Cinit)

testMe(); 

k = getK(Cinit');
ids = zeros(size(X, 2), 1);
[C, ids] = recursive(Cinit', X', ids, k);
fprintf('Overall distance %d\n', distortionFunction(X', ids, C, k));

A = ids';
C = C';

end

function testMe()
	Xi = [1, 1; 1, 3; 7, 3; 7, 1];
	X = [0, 1; 10, 7];
	ids = [0; 0; 0; 0];

	[actualC, actualIds] = recursive(X, Xi, ids, 2);
	expectedC = [1,2; 7,2];
	expectedIds = [1; 1; 2; 2];
	assert(actualC == expectedC);
	assert(actualIds == expectedIds);

	X = [0, 1; 10, 7; 12, 8];
	actualK = getK(X);
	expectedK = 3;
	assert(actualK == expectedK);

	A = [1, 1];
	B = [2, 2; 3, 3];

	difference = bsxfun(@minus, B, A);
	squares = difference .^ 2;
	result = sum(squares, 2);
	assert(result == [2;8]);

	X = [2, 2; 3, 3; 4, 4; 5, 5];
	C = [1, 1; 6, 6];
	ids = [1; 1; 1; 2];
	k = 2;
	actualDistance = distortionFunction(X, ids, C, k);
	expectedDistance = 2 + 8 + 18 + 2;
	assert(actualDistance == expectedDistance);
endfunction

function k = getK(X)
	k = size(X, 1);
endfunction

%ids maps points in X to points in C
function distance = distortionFunction(X, ids, C, k)
	distance = 0;
	for i = 1:k
		%points belonging to ci 
		pointsMappedToCi = X(ids==i, :);
		%ci
		ci = C(i, :);

		%calculating sum of ||(x - ci)||^2
		difference = bsxfun(@minus, pointsMappedToCi, ci);
		squares = difference .^ 2;
		result = sum(squares, 2);
		distance += sum(result);
	endfor
endfunction

function [C, ids] = recursive(X, Xi, oldIds, k)
	C = zeros([k, 2]);
	ids = dsearchn(X, Xi);
	for i = 1:k 
		cs = Xi(ids==i, :);
		c = [mean(cs(:, 1)), mean(cs(:, 2))];
		C(i,:) = c;
	endfor

	if not(isequal(ids, oldIds))
		[C, ids] = recursive(C, Xi, ids, k);
	end
endfunction

