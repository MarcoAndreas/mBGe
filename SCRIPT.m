
% SUPPLEMENTARY MATERIAL - MATLAB CODE

% for the paper
% ''Being Bayesian about Learning Bayesian Networks from Hybrid Data''
% International Journal of Approximate Reasoning

% MATLAB implementation of the MCMC algorithm
% to infer the mBGe model from data

% Code written by
% M. Grzegorczyk
% University of Groningen
% Netherlands (NL)
% Email: m.a.grzegorczyk@rug.nl

% Go to the directory where the MATLAB functions have been saved.
% For example:
cd X:\mBGe

% Set random seed:
rng(2025); % 2025 

% Number of MCMC iterations 
% 20,000 iterations will take a few minutes.
% Set to higher values for important applications (see paper).
n_iterations = 20000;

% Initial value of lambda
lambda = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate example data from a toy network
% with n=6 continuous and m=3 discrete (binary) variables
% 
% Continuous edges:
% X1 -> X3 
% X2 -> X3
% X3 -> X4
% X5 -> X6
% Discrete edges:
% Z1 -> X1
% Z3 -> X3

% True DAG among the continuous variables.
TRUE_DAG = zeros(6,6);

TRUE_DAG(1:2,3) = 1;
TRUE_DAG(3,4)   = 1;
TRUE_DAG(5,6)   = 1;

% Extract the true CPDAG.
[TRUE_CPDAG] = compute_CPDAG(TRUE_DAG);

% In the CPDAG, the edge X5->X6 is undirected (interpreted as bi-directional)
TRUE_CPDAG % Display the true CPDAG.

% Number of observations.
N = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate binary discrete data.
% The lowest level must be 1.

Z1 = randi([1 2], 1, N);
Z2 = randi([1 2], 1, N);
Z3 = randi([1 2], 1, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate continuous data with zero means.
% Impose the desired dependencies among the mean-adjusted variables.

X1 = randn(1, N);
X2 = randn(1, N);

% X1 -> X3 <- X2
X3 = 0.6*X1  + 0.5*X2 + 0.4*randn(1, N);

% X3 -> X4
X4 = 0.6*X3 + 0.1*randn(1, N);

X5 = randn(1,N);

% X5 -> X6
X6 = 0.7*X5 + 0.1*randn(1, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add two means via additive effects of the discrete variables.

% Z1 -> X1
X1 = X1 + 0.4*(Z1-1);

% Z3 -> X3
X3 = X3 + -0.3*(Z3-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_con = [X1; X2; X3; X4; X5; X6];

data_dis = [Z1; Z2; Z3];

% Number of levels of the discrete variables.
vec_dis = max(data_dis'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run the MCMC algorithm.
[BMA_1, BMA_2] = MCMC_mBGe(lambda, data_con, data_dis, vec_dis, n_iterations);

% Display the edge scores.

BMA_1

BMA_2

% Interpretation:
% BMA_1(i,j)
% is the score of the edge Xi -> Xj

% BMA_2(i,j)
% is the score of the edge Zi -> Xj

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the area under the precision-recall curve (AUPRC) 
% for the edges among the continuous nodes. 
auprc = compute_AUPRC(TRUE_CPDAG,BMA_1);

auprc

% Impose threshold psi on the scores of the continuous edges.
psi = 0.5;

% Predicted CPDAG.
BMA_1_psi = double(BMA_1>psi);

BMA_1_psi % Display predicted CPDAG.

% Compute relative structural Hamming distance (rSHD).
[rSHD] = compute_rSHD(TRUE_CPDAG,BMA_1_psi);

rSHD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Just for illustration purposes.
%
% Unrealistic (over-optimised) alternative threshold.
psi2 = 0.6396;

% New predicted CPDAG.
BMA_1_psi2 = double(BMA_1 > psi2);

% Compute relative structural Hamming distance (rSHD).
rSHD2 = compute_rSHD(TRUE_CPDAG, BMA_1_psi2);

% The relative structural Hamming distance (rSHD) is zero.
rSHD2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Now you can analyze your own data.
%
% You need an n-by-N data matrix 'data_con' with continuous values.
% data_con(i, j) is the j-th observation of the continuous node X_i.
%
% You need an m-by-N data matrix 'data_dis' with discrete values.
% data_dis(i, j) is the j-th observation of the discrete node Z_i.
%
% Important: The discrete levels must be integers starting at 1.
%
% The computational cost (runtime) increases with n_iterations.
% However, n_iterations needs to be large enough so that independent
% MCMC simulations yield very similar edge scores.
%
% END OF SCRIPT FILE.
