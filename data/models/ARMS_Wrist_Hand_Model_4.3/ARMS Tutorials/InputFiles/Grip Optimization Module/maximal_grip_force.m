import org.opensim.modeling.* % Import OpenSim modeling classes
%Copyright (c) 2021-present, Northwestern University, Shirley Ryan AbilityLab, Drexel University, University of Florida, and Edward Hines VA Medical Center. 
%All rights reserved. -------------------- The ARMS hand and wrist model has been open sourced solely for non-commercial purposes (including research, academic, 
%evaluation and personal uses) under the BSD 3-Clause License below. By downloading or using this software, (1) you accept the terms and conditions of the aforementioned
%open source license, (2) acknowledge that your use of this software is non-commercial and commercial use requires a commercial license, and (3) accept that use of the 
%model software must be acknowledged in all publications, presentations, or documents describing work in which the ARMS hand and wrist modell is used by citing the 
%following work: McFarland DC, Binder-Markey BI, Nichols JA, Wohlman SJ,de Bruin M, Murray WM. A Musculoskeletal Model of the Hand and Wrist Capable of Simulating 
%Functional Tasks. 2021; bioRxiv, p. 2021.12.28.474357, 2021, doi: 10.1101/2021.12.28.474357. 
%--------- Copyright (c) 2021-present, Northwestern University, Shirley Ryan AbilityLab, Drexel University, University of Florida, and Edward Hines VA Medical Center.
%All rights reserved. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: 
%1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 2. Redistributions in binary form must 
%reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
%3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific 
%prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
%BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
%CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
%GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
%TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	
rng('shuffle') %randomized the random number generator that sets initial activations
%Set up optimization
act_min=0.00; %minimum activation is 0
act_max=1.00; %maximum activation is 1

% Control bounds must be specified as a row vector, every column
% corresponds to a control; we have 15 independent actuators in the grip
% simulation
lowerBound = act_min*ones(1,15); 
upperBound = act_max*ones(1,15);

for i=1:1 % update the number of simulations you'd like to run
    initialCoefficients=rand([1,15]); % create initial random activations
    % Set up optimization parameters 
    options=optimoptions('simulannealbnd','Display','iter','DisplayInterval',1,'FunctionTolerance',.1,'StallIterLimit', 75);
    % Run the optimization: objective function is
    % GripForceObjectiveFunction
    [f,fval,exitflag,output] = simulannealbnd(@(coeffs0) GripForceObjectiveFunction(coeffs0),initialCoefficients,lowerBound,upperBound,options);
    % Display optimization results each iteration
    disp(strcat('Optimization #iterations=',num2str(output.iterations),', Grip Force = ',num2str(fval)));
    FVAL(i)=fval;% row vector of controls that produce the optimal result
    FVEC(i,:)=f; % objective function value: Grip strength - wrist penalty
    save('Results.mat');   % save results file after an iteration in case simulation crashs 
end