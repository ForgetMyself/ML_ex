function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

diff = X*theta-y;
J = (diff' * diff +lambda * theta(2:end)'*theta(2:end))/2/m; % 正则化时不考虑theta0
grad = X' * diff/m + lambda * theta / m; % 求梯度，先不考虑j=0
grad(1) = grad(1) - lambda*theta(1)/m;   % 将第一个梯度值减去正则化项，得到j=0时的梯度

% =========================================================================

grad = grad(:);

end
