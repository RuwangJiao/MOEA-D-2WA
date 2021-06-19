classdef MOEAD2WA < ALGORITHM
% <multi/many> <real/binary/permutation> <constrained>
% MOEA/D-2WA
% cp --- 4 --- Decrease trend of the dynamic constraint boundary

%------------------------------- Reference --------------------------------
% R. Jiao, S. Zeng, C. Li, and Y. S. Ong, Two-type Weight Adjustments in 
% MOEA/D for Highly Constrained Many-objective Optimization.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao  
    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            cp = Algorithm.ParameterSet(4);
            
            %% Generate the weight vectors
            OrN            = Problem.N;
            [W, Problem.N] = WeightGeneration(Problem.N, Problem.N, Problem.M, 1.0);

            %% Detect the neighbours of each solution
            T      = 20;
            nr     = 2;
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1:T);
            
            %% Generate random population
            Population    = Problem.Initialization();
            Z             = min(Population.objs, [], 1);
            [initialE, ~] = max(max(0, Population.cons), [], 1);
            initialE(initialE < 1) = 1;
            nCon          = size(Population.cons, 2);
            CV            = min(sum(max(0, Population.cons)./initialE, 2)./nCon, [], 1);
            Z             = [Z, CV]; 
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Reduce the dynamic constraint boundry
                epsn = ReduceBoundary(initialE, Problem.FE, Problem.maxFE - OrN, cp);
                
                % Update weights
                a    = (W(:, Problem.M + 1) - 1e-6) > epsn(:, 1)./initialE(:, 1);
                W(a, :) = [];
                
                % Update population
                if Problem.N > size(W, 1)
                    Problem.N  = size(W, 1);
                    B          = pdist2(W, W);
                    [~, B]     = sort(B, 2);
                    B          = B(:, 1:T);
                    Population = PopulationUpdate(Population, Problem.N, initialE, epsn, Z(:, 1:Problem.M));
                end
                % For each solution
                for i = 1 : Problem.N     
                    % Choose the parents
                    if rand < 0.9
                        P = B(i,randperm(size(B, 2)));
                    else
                        P = randperm(Problem.N);
                    end

                    % Generate an offspring
                    Offspring = OperatorGAhalf(Population(P(1:2)));

                    % Update the ideal point
                    Z(:, 1:Problem.M)   = min(Z(:, 1:Problem.M), Offspring.obj);
                    CV                  = min(sum(max(0, Offspring.cons)./initialE, 2)./nCon, [], 1);
                    Z(:, Problem.M + 1) = min(Z(:, Problem.M + 1), CV);

                    % Calculate the constraint violation of offspring and P
                    cvo = max(0, Offspring.con);
                    cvp = max(0, Population(P).cons); 
                    cvO = sum(max(0, Offspring.cons)./initialE, 2)./nCon;
                    cvP = sum(max(0, Population(P).cons)./initialE, 2)./nCon;

                    % Update the solutions in P by PBI approach
                    PObj    = [Population(P).objs, cvP];
                    OObj    = [Offspring.obj, cvO];
                    normW   = sqrt(sum(W(P, :).^2, 2));
                    normP   = sqrt(sum((PObj - repmat(Z, length(P), 1)).^2, 2));
                    normO   = sqrt(sum((OObj - Z).^2, 2));
                    CosineP = sum((PObj - repmat(Z, length(P), 1)).*W(P, :),2)./normW./normP;
                    CosineO = sum(repmat(OObj - Z,length(P), 1).*W(P, :),2)./normW./normO;
                    g_old   = normP.*CosineP + 5*normP.*sqrt(1 - CosineP.^2);
                    g_new   = normO.*CosineO + 5*normO.*sqrt(1 - CosineO.^2);
                    % Neighbor solution replacement
                    index   = find((sum(cvo<=epsn, 2)==nCon&sum(cvp<=epsn, 2)==nCon&g_old>=g_new)|(sum(cvo<=epsn, 2)==nCon&sum(cvp<=epsn, 2)<nCon)|(sum(cvo<=epsn,2)<nCon&sum(cvp<=epsn, 2)<nCon&sum(max(0, Population(P).cons), 2)>sum(max(0, Offspring.con))), nr);
                    Population(P(index)) = Offspring;
                end   
            end  
        end
    end
end