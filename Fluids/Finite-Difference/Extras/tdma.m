% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com
% Created:  Sat 12 Oct 2024 @ 14:41:41 +0200
% Modified: Tue 04 Mar 2025 @ 19:32:12 +0100

function phi = tdma(a, b, c, d)
	% Tridiagonal matrix solver
	N = length(d);
	
	% Forward elimination
	for i = 2:(N-1)
		w = a(i) / b(i-1);
		b(i) = b(i) - w * c(i-1);
		d(i) = d(i) - w * d(i-1);
	end
	
	% Backwards substitution
	phi = zeros(N, 1);
	phi(N) = d(N) / b(N);
	for i = (N-1):-1:1
		phi(i) = (d(i) - c(i) * phi(i+1)) / b(i);
	end
end
