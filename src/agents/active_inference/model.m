function mdp = model

clear;
clc;
% Initial State: P(s_0):
D{1} = [128 zeros(1,8)]';                % Location:
D{2} = [2 2]';                                   % Context: 

% P(o|s)
Nf = numel(D); 

% Number of factor in each state
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end

No    = [9 3];    % Reward [Positive, Negative, Neutral]
                
% Total outcome levels: 3
Ng    = numel(No); 

% Matrix with all possible combinations (outcomes;states)'
e = 0;
for g = 1:Ng
    A{g} = ones([No(g),Ns])*e; 
end
          
env1 =  ['S' 'F' 'F' 'F' 'F' 'G' 'F' 'H' 'F'];
env2 =  ['S' 'F' 'F' 'F' 'F' 'H' 'F' 'G' 'F'];

for f1 = 1:Ns(1) 
    for f2 = 1:Ns(2) 
                                    A{1}(f1, f1,:) = 1;

                                    if f2 == 1
                                        env = env1;
                                    else
                                        env = env2;
                                    end
                                    
                                    % Adding the slippery part:
                                    if env(f1) == 'H'
                                        A{2}(2,f1,f2) = 1;
                                    % Adding the goal location:
                                    elseif  env(f1) == 'G'
                                        A{2}(1,f1,f2) = 1; 
                                    else 
                                        A{2}(3,f1,f2) =1;
                                    end  
    end
end

m = 1000;
for g = 1:Ng
    a{g} = A{g}*m;
end


%--------------------------------------------------------------------------
% Transitions from t to t+1: P(S_t| S_t-1, pi)
% The B(u) matrices encode action-specific transitions
%--------------------------------------------------------------------------
e = 0;
for f = 1:Nf
        B{f} = e*ones(Ns(f));
end

% Actions = 4, left, down, right, up:
% Left:
B{1}(:,:,1)  =  circshift(eye(9)*1 + e, -1);
for i = [1, 4, 7]
    B{1}(:, i,1) = e;
    B{1}(i, i,1) = 1+e;
end

% down:
B{1}(:,:,2) = e*ones(Ns(1));
for i = 1:5
    B{1}(i+3, i,2) = 1+ e;
end
for i = 7:9 
    B{1}(:, i,2) = e;
    B{1}(i,i,2) = 1+e;
end

% Right: 
B{1}(:,:,3)  = circshift(eye(9)*1 + e, 1);
for i = [3,6, 9]
    B{1}(:, i,3) = e;
    B{1}(i, i,3) = 1+e;
end

% Up:
B{1}(:,:,4) = e*ones(Ns(1));
for i = 4:9
    B{1}(i-3, i,4) = 1+ e;
end
for i = 1:3 
     B{1}(:, i,4) = e;
    B{1}(i,i,4) = 1+e;
end


% Goal State: 
B{1}(:,6,1:4) = e; 
B{1}(6,6,1:4) = 1+e; 

B{1}(:,8,1:4) = e; 
B{1}(8,8,1:4) = 1+e; 


% Context, which cannot be changed by action
%--------------------------------------------------------------------------
B{2}(:,:,1)  = eye(2)*1;

for g = 1:Nf
    b{g} = B{g}*m;
end


% ln(P(o))
for g = 1:Ng
    C{g}  = zeros(No(g),1);
end

c= 4;
C{1}(1,:) = -c;

C{2}(1,:) = c;
C{2}(2,:) = -c;



%------------------------------------------------------------------------------
% Allowable policies (of depth T).  These are just sequences of actions
% Time * Poilicy * Factor
%------------------------------------------------------------------------------

% Policies:

V(:,:,1) =  [2 3 1 4 2 2 3 3 3 ;  
                 2 3 1 4 2 3 3 2 2 ;
                 2 3 1 4 3 2 2 2 3 ];  
                
V(:,:,2) =  1;

 
%--------------------------------------------------------------------------
% Specify the generative model
%--------------------------------------------------------------------------

mdp.alpha = 2048;
mdp.A = A;
mdp.B = B; 
mdp.C = C;   
mdp.D = D;                      
mdp.s = [1,1]';
mdp.V =V; 
mdp.tau = 12;

% Labels:
mdp.Bname = {'Location','Context'};
mdp.Aname = {'Position-A','Goal-A'};
mdp.label.modality = {'Position','Goal'};

% Checking model:
%mdp         = spm_MDP_check(mdp);

return 

