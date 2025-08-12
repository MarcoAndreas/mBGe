function [BMA_1, BMA_2] = MCMC_mBGe(lambda, data_con, data_dis_raw, vec_dis, n_iterations)

a_hyper = 0.1;
b_hyper = 0.1;

[S,T] = size(data_con);
% There are S continuous variables 
% and T observations (each S-dimensional)

[N,~] = size(data_dis_raw);
% There are N discrete variables (and T observations)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = 1*(S+2); 

V_Mat = (alpha-S-1)*eye(S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% We build the full design matrix for an additive model

% Loop through the discrete nodes

data_dis = [];

MAPPING = zeros(N,sum(vec_dis-1));


offset_val = 0;

for i=1:N

    v_dis = vec_dis(i); % no of values of discrete node no. i

    inter = offset_val + (1:(v_dis-1));

    MAPPING(i,inter) = 1;
   
    offset_val = offset_val + (v_dis-1);

    for j=2:v_dis

        new_col = zeros(T,1);
        new_col(find(data_dis_raw(i,:)==j),1) = 1;

        if(sum(new_col)>0)
            data_dis = [data_dis,new_col]; % (v_dis-1)-by- T
        end

    end
end


data_dis = data_dis';


% data_dis is N-by-T


% MAPPING is N-by-sum(vec_dis-1)


% EDGE INTERACTION SCORES
BMA_1 = zeros(S,S);
BMA_2 = zeros(N,S);

Counter     = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialisation

% Empty error DAG 
GRAPH    = zeros(S,S);
ANCESTOR = expm(GRAPH) - eye(S);

% Empty covariate matrix
COV_SYS = zeros(N,S);

%%%%%%%% MODEL PARAMETERS

% Reg. coefficient vector
% There are always S intercept parameters.
BETA = zeros(S,1);


for i=1:n_iterations
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sampling step 1: MH-move on graph
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fan_in  = S-1; % by default there is no fan-in restriction

    % Select a random neighbour DAG
    [GRAPH_CANDIDATE, op, nodes, n_neighbours_old] = neighbour_random(GRAPH, ANCESTOR, fan_in);

    % Compute the new ANCESTOR MATRIX and the no. of neighbour graphs of GRAPH_CANDIDATE
    ANCESTOR_CANDIDATE       = update_ancestor_matrix(ANCESTOR, op, nodes(1), nodes(2), GRAPH_CANDIDATE);
	[n_neighbours_candidate] = n_neighbours(GRAPH_CANDIDATE,ANCESTOR_CANDIDATE, fan_in);
    
    % SUBTRACT REGRESSION EXPECTATIONS   
    [DATA] = MAKE_NEW_RESPONSES(data_con,data_dis, MAPPING, COV_SYS, BETA);

    [log_score_candidate] = COMPUTE_LOCAL_GRAPH_LOG_SCORE(DATA, GRAPH_CANDIDATE, nodes(2), V_Mat, alpha);
    [log_score_old]       = COMPUTE_LOCAL_GRAPH_LOG_SCORE(DATA, GRAPH,           nodes(2), V_Mat, alpha);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (op=='rev')
        log_score_candidate = log_score_candidate + COMPUTE_LOCAL_GRAPH_LOG_SCORE(DATA, GRAPH_CANDIDATE, nodes(1), V_Mat, alpha);
        log_score_old       = log_score_old       + COMPUTE_LOCAL_GRAPH_LOG_SCORE(DATA, GRAPH,           nodes(1), V_Mat, alpha); 
    end
  
    Accept_prob = exp(log_score_candidate - log_score_old) * (n_neighbours_old/n_neighbours_candidate);
    
    u = rand;
         
    if (u<=Accept_prob)
        GRAPH      = GRAPH_CANDIDATE;
        ANCESTOR   = ANCESTOR_CANDIDATE;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sampling step 2: Gibbs-move on covariance matrix 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    SCALE_MAT  = inv(inv(V_Mat) + DATA * DATA');
    W_FULL     = wishrnd(SCALE_MAT,alpha+T);
    SIGMA_FULL = inv(W_FULL);
    
    % Make covariance matrix consistent with current graph
    [SIGMA]    = EXTRACT_PRECISION_OF_DAG_NEW(SIGMA_FULL,GRAPH);

    SIGMA_inv = inv(SIGMA);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sampling step 3: MH-move on covariates 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    COV_SYS_CANDIDATE = COV_SYS;
    
    % Randomly select a response and a covariate
    x = randi([1,N]); % random covariate
    y = randi([1,S]); % random response
    
    % Delete/add covariate x for response y
    COV_SYS_CANDIDATE(x,y) = 1 - COV_SYS_CANDIDATE(x,y); 
    
    % Compute marginal likelihoods (marginalized over BETA)
    [log_score_candidate] = COMPUTE_REGRESSION_LOG_SCORE(data_con, data_dis, MAPPING, SIGMA_inv, COV_SYS_CANDIDATE, lambda);
    [log_score_old]       = COMPUTE_REGRESSION_LOG_SCORE(data_con, data_dis, MAPPING, SIGMA_inv, COV_SYS          , lambda);

    Accept_prob = exp(log_score_candidate - log_score_old);
    
    u = rand;
         
    if (u<=Accept_prob)
        COV_SYS      = COV_SYS_CANDIDATE;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sampling step 4: Gibbs-move on regression coefficients 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    DATA_dis_extended = [ones(1,T);data_dis];
    COV_SYS_extended  = [ones(1,S);COV_SYS];

    [N_val,n_val] = size(MAPPING);

    MAPPING_extended = zeros(N_val+1,n_val+1);
    MAPPING_extended(1,1) = 1;
    MAPPING_extended(2:end,2:end) = MAPPING;


    ind_all = [];

    VALUES_PER_NODE = zeros(1,S);

    for s=1:S
   
        parents_s    = find(COV_SYS_extended(:,s));
        VEC          = sum(MAPPING_extended(parents_s,:));
        covariates_s = find(VEC);

        ind_all = [ind_all,covariates_s]; % collect parents of s   

        VALUES_PER_NODE(1,s) = length(covariates_s);
    
    end


    % Total no. of regression coefficients (+S intercept parameters)

    n_k = length(ind_all);


    LARGE_MAT = DATA_dis_extended(ind_all,:)'; 
    CELL_DATA = mat2cell(LARGE_MAT,ones(1,T),VALUES_PER_NODE); % column-sums
    
    % Initialisation:
    SUM_X = zeros(n_k,n_k);
    SUM_Y = zeros(n_k,1);
    

    for t=1:T      
        X_t = blkdiag(CELL_DATA{t,:});
        
        SUM_X = SUM_X + X_t' * SIGMA_inv * X_t;
        SUM_Y = SUM_Y + X_t' * SIGMA_inv * data_con(:,t);
    end
    
    % SIGMA_0 = lambda * eye(k);

    inv_SIGMA_0 = (1/lambda) * eye(n_k);	

    R        = chol(inv_SIGMA_0 + SUM_X); 
    R_inv    = R\eye(n_k);   
    BETA_COV = R_inv * R_inv';
        
    BETA_MEAN = BETA_COV * SUM_Y;

    % Sample new regression coefficient vector from full conditional
    BETA = mvnrnd(BETA_MEAN,(BETA_COV+BETA_COV')/2)';



    a_post = a_hyper + length(BETA)/2;
    b_post = b_hyper + sum(BETA.^2)/2;


    lambda = 1/gamrnd(a_post,1/b_post);

 
     % After burn in phase (50%) start to include samples
    if (i>(n_iterations/2))

      [OUT] = dag_to_cpdag(GRAPH);
      
      [ind1_c] = find(OUT==2); % compelled edge
      [ind1_r] = find(OUT==3); % reversible edge
    
      DAG1 = zeros(S,S);
      DAG2 = zeros(S,S);
    
      DAG1(ind1_c) = 1;
      DAG2(ind1_r) = 1; % reversible edge
  
 
      BMA_1 = BMA_1 + DAG1 + DAG2 + DAG2';
      BMA_2 = BMA_2 + COV_SYS;
	  
      Counter = Counter + 1;

    end	
end

BMA_1 = BMA_1/Counter; % continuous -> continuous
BMA_2 = BMA_2/Counter; % discrete   -> continuous
  
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [log_score] = COMPUTE_REGRESSION_LOG_SCORE(DATA_con, DATA_dis, MAPPING, SIGMA_inv, COV_SYS, lambda)

[S,T] = size(DATA_con);

% Build one single long vector of response values
RESPONSE_VEC = reshape(DATA_con,S*T,1);

% Expectation of marginal Gaussian
MU_VEC       = zeros(S*T,1);

DATA_dis_extended = [ones(1,T);DATA_dis];
COV_SYS_extended  = [ones(1,S);COV_SYS];

[N_val,n_val] = size(MAPPING);

MAPPING_extended = zeros(N_val+1,n_val+1);
MAPPING_extended(1,1) = 1;
MAPPING_extended(2:end,2:end) = MAPPING;


ind_all = [];

VALUES_PER_NODE = zeros(1,S);

for s=1:S
   
    parents_s    = find(COV_SYS_extended(:,s));
    VEC          = sum(MAPPING_extended(parents_s,:));
    covariates_s = find(VEC);

    ind_all = [ind_all,covariates_s]; % collect parents of s   

    VALUES_PER_NODE(1,s) = length(covariates_s);
    
end

% Total no. of regression coefficients (+S intercept parameters)

n_k = length(ind_all);

LARGE_MAT = DATA_dis_extended(ind_all,:)'; 
CELL_DATA = mat2cell(LARGE_MAT,ones(1,T),VALUES_PER_NODE); % column-sums

for t=1:T 
    X_NEW{t} = blkdiag(CELL_DATA{t,:});
    SIGMA_inv_X_NEW{t,1}  = SIGMA_inv  * X_NEW{t};
    X_NEW_SIGMA_inv{1,t} = X_NEW{t}' * SIGMA_inv;
end


SIGMA_inv_X_NEW = (RESPONSE_VEC-MU_VEC)' * cell2mat(SIGMA_inv_X_NEW);
X_NEW_SIGMA_inv = cell2mat(X_NEW_SIGMA_inv) * (RESPONSE_VEC-MU_VEC);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SIGMA_0_inv = (1/lambda) * eye(n_k);

X_LARGE_BLOCK_MAT_X_2 = zeros(n_k,n_k);

for t=1:T
    X_LARGE_BLOCK_MAT_X_2 = X_LARGE_BLOCK_MAT_X_2 + X_NEW{t}' * SIGMA_inv * X_NEW{t};
end

MAT = SIGMA_0_inv + X_LARGE_BLOCK_MAT_X_2; % (nk)-by-(nk)  + (nk)-by-(T*S) * (T*S)-by-(nk)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n_Sig,~]   = size(MAT); 
R           = chol(MAT); 
R_inv       = R\eye(n_Sig);   
INVERSE_MAT = R_inv * R_inv';  % (nk)-by-(nk)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

log_det = log(det(INVERSE_MAT)) + (T*log(det(SIGMA_inv)) + S*log(1))  +     log(det(MAT - X_LARGE_BLOCK_MAT_X_2));

SUM = 0;

for t=1:T
    SUM = SUM + DATA_con(:,t)' * SIGMA_inv * DATA_con(:,t); 
end

log_score = -length(RESPONSE_VEC)/2 * log(2*pi) + 0.5 * log_det  -0.5 *   SUM   +0.5 * SIGMA_inv_X_NEW *  INVERSE_MAT * X_NEW_SIGMA_inv;


return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [SIGMA_NEW] = EXTRACT_PRECISION_OF_DAG_NEW(SIGMA,DAG)

n = length(SIGMA);

% To be filled:
V_VEC = zeros(n,1);
B_MAT = zeros(n,n);

for i=1:n
   
    parents_i = find(DAG(:,i));
    
    if (~isempty(parents_i))
        
        SIGMA_parents     = SIGMA(parents_i,parents_i);
        R                 = chol(SIGMA_parents); 
        R_inv             = R\eye(length(parents_i));   
        SIGMA_parents_inv = R_inv * R_inv';
        
        B_MAT(parents_i,i) =             (SIGMA(i,parents_i) * SIGMA_parents_inv)';
        V_VEC(i,1)         = SIGMA(i,i) - SIGMA(i,parents_i) * SIGMA_parents_inv * SIGMA(parents_i,i);
    else
        V_VEC(i,1) = SIGMA(i,i);
    end
    
end

[order_DAG] = topological_sort(DAG);

[~,indicis] = sort(order_DAG); % To be able to re-order to the original variable order 


V_VEC = V_VEC(order_DAG);
B_MAT = B_MAT(order_DAG,order_DAG);


[W] = COMPUTE_PRECISION_MATRIX_NEW(V_VEC,B_MAT);

W         = W(indicis,indicis);
R         = chol(W); 
R_inv     = R\eye(n);   
SIGMA_NEW = R_inv * R_inv';



% P(X_i|X_1,...X_{i-1}) = N(m_i + sum_j b_{ij} * (x_j-m_j), 1/v_i)

% Conditional Gaussian X_i|(X_1,...X_{i-1}) ~ N(mu,sigma^2)

% vec = (X_1,...,X_{i-1})
% mu      = E[X_i] + Sigma_{i,vec} * Sigma_{vec,vec}^{-1} * (obs_vec - E[vec])
% sigma^2 = Var(X_1) -  Sigma_{i,vec} * Sigma_{vec,vec}^{-1} * Sigma_{vec,i}


% B_MAT STRUCTURE

% 0  b12  b13  b14
% 0   0   b23  b24
% 0   0    0   b34

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W] = COMPUTE_PRECISION_MATRIX_NEW(V_VEC,B_MAT)

% P(X_i|X_1,...X_{i-1}) = N(m_i + sum_j b_{ij} * (x_j-m_j), 1/v_i)

% m_i is the unconditional mean of x_i
% v_i is the variance of x_i given x1,...,x_{i-1}

% Conditional Gaussian X_i|(X_1,...X_{i-1}) ~ N(mu,sigma^2)

% vec = (X_1,...,X_{i-1})
% mu      = E[X_i] + Sigma_{i,vec} * Sigma_{vec,vec}^{-1} * (obs_vec - E[vec])
% sigma^2 = Var(X_1) -  Sigma_{i,vec} * Sigma_{vec,vec}^{-1} * Sigma_{vec,i}


% B_MAT STRUCTURE

% 0  b12  b13  b14
% 0   0   b23  b24
% 0   0    0   b34

n = length(V_VEC);

W = 1/V_VEC(1);

for i=1:(n-1)
    
    W = W + (B_MAT(1:i,i+1) * B_MAT(1:i,i+1)')/V_VEC(i+1);
    
    new_vec = (-1) * B_MAT(1:i,i+1)/V_VEC(i+1);
   
    W = [[W,new_vec];[new_vec',1/V_VEC(i+1)]];
 
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [DATA_con_NEW] = MAKE_NEW_RESPONSES(DATA_con,DATA_dis,MAPPING,COV_SYS,BETA)

% COV_SYS is N-by-S

[S,T] = size(DATA_con);

DATA_con_NEW = zeros(S,T);

index_start = 1;

for s=1:S
 
    parents_s = find(COV_SYS(:,s));
    VEC = sum(MAPPING(parents_s,:));
    covariates_s = find(VEC);


    index_end    = index_start +length(covariates_s);
    for t=1:T
        DATA_con_NEW(s,t) = DATA_con(s,t) - [1;DATA_dis(covariates_s,t)]' * BETA(index_start:index_end); 
    end
    index_start = index_end+1;
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [log_score] = COMPUTE_LOCAL_GRAPH_LOG_SCORE(DATA, GRAPH, s_node, V_Mat, alpha )

% S nodes and T observations
[S,T] = size(DATA);

parents_of_node  = find(GRAPH(:,s_node));

GRAPH(s_node,s_node) = 1;
parents_and_node = find(GRAPH(:,s_node));
            
T_0 = inv(V_Mat);
T_m = T_0 + DATA * DATA';

[log_score_1] = Gauss_Score_complete(S, length(parents_and_node), T, alpha, T_0(parents_and_node,parents_and_node), T_m(parents_and_node,parents_and_node));
[log_score_2] = Gauss_Score_complete(S, length(parents_of_node),  T, alpha, T_0(parents_of_node, parents_of_node),  T_m(parents_of_node ,parents_of_node));
                 
log_score = log_score_1 - log_score_2;
  
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [log_score] = Gauss_Score_complete(S, n_nodes_in_sub, n_obs, alpha, T_0_sub, T_m_sub)

if (n_nodes_in_sub==0) % For an empty parent set
    log_score = 0;
else
      
    sum_1 = ((-1)*(n_nodes_in_sub)*(n_obs)/2) * log(pi);

    sum_2 = ((alpha - S+n_nodes_in_sub)/2) * log(det(T_0_sub)) + (-1)*(alpha+n_obs - S+n_nodes_in_sub)/2 * log(det(T_m_sub));

    sum_3 = logmvgamma(alpha+n_obs-S+n_nodes_in_sub,n_nodes_in_sub) - logmvgamma(alpha-S+n_nodes_in_sub,n_nodes_in_sub);  
   
    log_score = sum_1 + sum_2 + sum_3; 

end

return 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = logmvgamma(x,d)

% Compute logarithm multivariate Gamma function.

% Gamma_p(x) = pi^(p(p-1)/4) prod_(j=1)^p Gamma(x+(1-j)/2)

% log Gamma_p(x) = p(p-1)/4 log pi + sum_(j=1)^p log Gamma(x+(1-j)/2)

% Written by Michael Chen (sth4nth@gmail.com).

s = size(x);

x = reshape(x,1,prod(s));

x = bsxfun(@plus,repmat(x/2,d,1),(1-(1:d)')/2); % CORRECTED x <- x/2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = d*(d-1)/4*log(pi)+sum(gammaln(x),1);

y = reshape(y,s);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Nachbar, op, nodes, graph_neighbours] = neighbour_random(G0, A, fan)

n_nodes = length(G0); 

SS=sum(G0); % parent-set-size of each network-node (column-sums)

SS=(SS>=fan); % Nodes already having a parent-set of fan-in size (indicator-vector)
SS=find(SS); % The indicis of these nodes 
             
% EDGE-DELETIONS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[I,J] = find(G0); % indicis of existing edges
E = length(I);    % number of exisiting edges

% EDGE-REVERSALS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = max(0, G0-(G0'*A)'); % L(i,j)=1 if and only if X_i->X_j is reversible 

L(SS,:)=0; % edges pointing from nodes with maximal fan-in are not reversible  
           
[IL, JL] = find(L);  % indices of reversible edges 

EL = length(IL); % number of reversible (non covered) 

% EDGE-ADDITIONS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gbar = ~G0;  % Alle non-existing edges 
% exclude: self-loops
self_loops = repmat(0, 1, n_nodes); 
diags = 1:n_nodes+1:n_nodes^2;
% Von 1 bis n^2 in (n+1)er-Schritten 
Gbar(diags) = self_loops;

GbarL = Gbar-A; % avoid cycles 
GbarL(:,SS)=0;  % exclude edges leading to nodes with maximal fan-in
            
[IbarL, JbarL] = find(GbarL);  % indicis of edges that can be added 
EbarL = length(IbarL); % number of these edges

graph_neighbours = (E+EL+EbarL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Randomly chose a neighbour graph:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Bestimme zunaechst, die Operation: del, rev oder add
Probs=[E, EL, EbarL]/(graph_neighbours);

R=rand(1); 

% Bestimme die Verteilungsfunktion von Probs:
cumprob = cumsum(Probs(:));
cumprob = cumprob(1:end-1);

% Wieviele Werte der Verteilungsfunktion werden ueberschritten?
Zufall = sum(R > cumprob)+1;
      
% Beruecksichtigt wurde, dass durch die drei Operationen 
% jeweils unterschiedlich viele Nachbar-Graphen erreicht werden koennen!
% Die Variable "Zufall" ist aus der Menge {1,2,3}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Schreibe die Adjacency-Matrix als Spaltenvektor:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gkopie = G0(:);

% In Abhaengigkeit von der gewaehlten Operation:

if Zufall==1 % EDGE-DELETION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
op='del';
Auswahl=random('Discrete Uniform',E,1,1);
% Waehle zufaellig eine der E Kanten, die entfernt werden kann
x = I(Auswahl); % Die Indices
y = J(Auswahl); % dieser Kante

ndx = subv2ind([n_nodes n_nodes], [x y]); % Berechne den 1-dim Index dieser Kante
% Fasse dazu [x y] als Element einer nxn-Matrix auf

Gkopie(ndx)=0; % Enferne die Kante
Nachbar = reshape(Gkopie, [n_nodes n_nodes]); % Forme die neue Adjacency-Matrix

elseif Zufall==2 % EDGE-REVERSAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
op='rev';
Auswahl=random('Discrete Uniform',EL,1,1);
x = IL(Auswahl); % Indices der Kante
y = JL(Auswahl); % die umgekehrt wird

ndx = subv2ind([n_nodes n_nodes], [x y]); 
rev_ndx = subv2ind([n_nodes n_nodes], [y x]); % die 1-dim Indices
Gkopie(ndx) = 0; % Kante entfernen
Gkopie(rev_ndx) = 1; % neue Kante hinzufuegen
Nachbar = reshape(Gkopie, [n_nodes n_nodes]); % wieder eine Adjacency-Matrix formen

else % EDGE-ADDITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
op='add';
Auswahl=random('Discrete Uniform',EbarL,1,1);
x = IbarL(Auswahl);
y = JbarL(Auswahl);
 
ndx = subv2ind([n_nodes n_nodes], [x y]);
Gkopie(ndx) = 1;
Nachbar = reshape(Gkopie, [n_nodes n_nodes]);
end

nodes=[x y];

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = update_ancestor_matrix(A,  op, i, j, Nachbar)

switch op
case 'add'
 A = do_addition(A, op, i, j, Nachbar);
case 'del' 
 A = do_removal(A, op, i, j, Nachbar);
case 'rev'
 A = do_removal(A, op, i, j, Nachbar);
 A = do_addition(A, op, j, i, Nachbar);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = do_addition(A, op, i, j, GRAPH_NEW)

A(j,i) = 1; 

all_ancestors_i = find(A(i,:)); % All ancestors of X_i 

if ~isempty(all_ancestors_i) % If this set of ancestors of X_i is non-empty:  
 A(j,all_ancestors_i) = 1;   % These nodes become ancestors of X_j too 
end

all_ancestors_j   = find(A(j,:)); %  All ancestors of X_j (after adding X_i and its ancestors)
all_descendents_j = find(A(:,j)); %  All descendents of X_j (before adding X_i)

if ~isempty(all_ancestors_j)    % If the set of ancestors of X_j is non-empty:
    
    for k=all_descendents_j(:)' % For each descendent of X_j
    A(k,all_ancestors_j) = 1;   % add to the ancestors of this descendent k the ancestors of X_j 
    end
    
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = do_removal(A, op, i, j, GRAPH_NEW)

descendents_of_j = find(A(:,j)); 
order            = topological_sort(GRAPH_NEW);

[THRASH, perm] = sort(order);

descendents_of_j_sorted = perm(descendents_of_j);

[THRASH, perm]   = sort(descendents_of_j_sorted);
descendents_of_j = descendents_of_j(perm);

% Update the descendents of X_j und fuer alle Nachfahren von Xj
A = update_row(A, j, GRAPH_NEW);

for k = descendents_of_j(:)'
A = update_row(A, k, GRAPH_NEW);
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = update_row(A, j, GRAPH_NEW)

% Determine the j-th row of A:
A(j, :) = 0; % Set this row = 0 
ps = find(GRAPH_NEW(:,j))'; % Determine the parents of X_j

if ~isempty(ps) % If X_j has parents
 A(j, ps) = 1; % All parents of X_j are andestors of X_j
end
for k=ps(:)' % For each parent k of X_j:
 ancestors_of_k = find(A(k,:)); % Determine the ancestors of k
 if ~isempty(ancestors_of_k)    % If there are some ancestors of k: 
   A(j, ancestors_of_k) = 1;    % set them ancestors of X_j
 end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function order = topological_sort(dag) 

n          = length(dag);
indeg      = zeros(1,n);
zero_indeg = [];          % a stack of nodes with no parents

for i=1:n
indeg(i) = length(find(dag(:,i)));
  
     if indeg(i)==0
     zero_indeg = [i zero_indeg];
     end
  
end

t=1;
order = zeros(1,n);

while ~isempty(zero_indeg)
  v = zero_indeg(1); % pop v
  zero_indeg = zero_indeg(2:end);
  order(t) = v;
  t = t + 1;
  cs = find(dag(v,:));
  
  for j=1:length(cs)
    c = cs(j);
    indeg(c) = indeg(c) - 1;
      if indeg(c) == 0
      zero_indeg = [c zero_indeg]; % push c n_nodes = n;
      end
  end
  
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ndx = subv2ind(siz, subv)

[ncases ndims] = size(subv);

if all(siz==2)
  twos = pow2(0:ndims-1);
  ndx = ((subv-1) * twos(:)) + 1;
else
  cp = [1 cumprod(siz(1:end-1))]';
  ndx = (subv-1)*cp + 1;
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [graph_neighbours] = n_neighbours(G0, A, fan)

n_nodes = length(G0); % number of network-nodes

SS=sum(G0);   % Parent-set-size of each network-node (column-sums)
SS=(SS>=fan); % Nodes already having a parent-set of fan-in size (indicator-vector)
SS=find(SS);  % The indicis of these nodes 
             
% EDGE-DELETIONS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[I,J] = find(G0); % indicis of existing edges
E = length(I);    % number of exisiting edges

% EDGE-REVERSALS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = max(0, G0-(G0'*A)'); % L(i,j)=1 if and only if X_i->X_j is reversible 

L(SS,:)=0; % edges pointing from nodes with maximal fan-in are not reversible  
           
[IL, JL] = find(L);  % indices of reversible edges 

EL = length(IL); % number of reversible (non covered) 

% EDGE-ADDITIONS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gbar = ~G0;  % Alle non-existing edges 
% exclude: self-loops
self_loops = repmat(0, 1, n_nodes); 
diags = 1:n_nodes+1:n_nodes^2;
% Von 1 bis n^2 in (n+1)er-Schritten 
Gbar(diags) = self_loops;

GbarL = Gbar-A; % avoid cycles 
GbarL(:,SS)=0;  % exclude edges leading to nodes with maximal fan-in      
[IbarL, JbarL] = find(GbarL);  % indicis of edges that can be added 
EbarL = length(IbarL); % number of these edges

graph_neighbours = E+EL+EbarL;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dagx] = dag_to_cpdag(dagx)

order = topological_sort(dagx); % get the topological order of nodes and their number


[nx,ny] = size(dagx); % gets the number of nodes, note that nx == ny

[I,J] = find(dagx); % finds all nonzero elements in the adjacency matrix, i.e. arcs in the DAG - however we will overwrite it in a special order

% we will sort the arcs from lowest possible y and highest possible x, arcs are x->y

e = 1;

for y = 1:ny

    for x = nx:-1:1

        %fprintf('x %d ',order(x)); fprintf('y %d ',order(y));

        if dagx(order(x),order(y)) == 1 

            I(e) = order(x);

            J(e) = order(y);

            e = e + 1;

        end

    end

end

% Now we have to decide which arcs are part of the essential graph and

% which are undirected edges in the essential graph.

% Undecided arc in the DAG are 1, directed in EG are 2 and undirected in EG

% are 3.

for e = 1:length(I)

    if dagx(I(e),J(e)) == 1

        cont = true;

        for w = 1:nx 

            if dagx(w,I(e)) == 2

                if dagx(w,J(e)) ~= 0

                    dagx(w,J(e)) = 2;

                else

                    for ww = 1:nx

                        if dagx(ww,J(e)) ~= 0

                           dagx(ww,J(e)) = 2;

                        end

                    end % and now skip the rest and start with another arc from the list

                    w = nx;

                    cont = false;

                end

            end

        end

        if cont

           exists = false;

           for z = 1:nx

               %fprintf('test %d',dagx(z,J(e)));

               if dagx(z,J(e)) ~= 0 & z ~= I(e) & dagx(z,I(e)) == 0

                  exists = true; 

                  for ww = 1:nx

                        if dagx(ww,J(e)) == 1

                           dagx(ww,J(e)) = 2;

                        end 

                  end

               end

           end

           if ~ exists

               for ww = 1:nx

                   if dagx(ww,J(e)) == 1

                      dagx(ww,J(e)) = 3;

                   end 

               end  

           end

        end

    end            

end

return



























