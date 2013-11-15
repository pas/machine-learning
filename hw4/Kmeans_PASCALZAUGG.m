function [C, A] = Kmeans_PASCALZAUGG(X, Cinit)

testMe(); 

k = getK(Cinit');
ids = zeros(size(X, 2), 1);
[C, ids] = recursive(Cinit', X', ids, k)

A = ids';

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
endfunction

function k = getK(X)
	k = size(X, 1);
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

