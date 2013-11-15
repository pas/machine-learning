function C = recursive(X, Xi, k)
	C = zeros([k, 2]);
	ids = dsearchn(X, Xi);
	for i = 1:k 
		cs = Xi(ids==i, :);
		c = [mean(cs(:, 1)), mean(cs(:, 2))];
		C(i,:) = c;
	endfor

	if isequal(C, Xi)
		C;
	else
		C = recursive(X, C, k);
	end
endfunction

Xi = [1, 1; 1, 3; 7, 3; 7, 1]
X = [0, 1; 10, 7]
recursive(X, Xi, 2)



