
function [true_cpdag] = COMPUTE_CPDAG(true_dag)

      [S,~] = size(true_dag);

      [OUT] = dag_to_cpdag(true_dag);
      
      [ind_c] = find(OUT==2); % compelled edge
      [ind_r] = find(OUT==3); % reversible edge
    
      DAG1 = zeros(S,S);
      DAG2 = zeros(S,S);
    
      DAG1(ind_c) = 1;
      DAG2(ind_r) = 1; % reversible edge
 
      true_cpdag = DAG1 + DAG2 + DAG2';

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


















