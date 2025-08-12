

function [shd] = compute_rSHD(CPDAG_TRUE,CPDAG)

SKELETON = CPDAG_TRUE + CPDAG_TRUE';
n_edges  = sum(sum((SKELETON>0)))/2;

MISS = (CPDAG_TRUE~=CPDAG);

MISS2 = MISS + MISS';

shd = sum(sum(MISS2>0))/2;

shd = shd/n_edges;

return


