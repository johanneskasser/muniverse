function f = GripForceObjectiveFunction(activations)
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

%GRIPFORCEOBJECTIVEFUNCTION: This is an objective function to optimize grip
%force from a computational OpenSim hand model. This function takes in
%activations for the individual muscles, updates the control file for a
%forward dynamic simulation, and then runs the forward dynamic simulation
%via a predefined setup file. The objective value is Grip force which is
%defined as the magnitude of the JRF between a massless body and the
%thridmc minus a penalty term for wrist motion

	% Import OpenSim modeling classes
	import org.opensim.modeling.*
    
    %Get model and muscle set
    myModel=Model('Grip_Model.osim'); % Get model
    MS=myModel.getMuscles(); % Get muscle set
    MS_size=MS.getSize(); % Get # of muscles 
    
    % Update controls file and model with current activations    
    controls = Storage('controls.sto');  % Load the controls file. 
    for i=1:38 % grip simulations have 38 muscle, we exclude 5 intrinsic thumb muscles 
        if i<7 % set activation for primary wrist muscles
            x=activations(1,i);
        elseif i==7||i==8||i==9||i==10 % Set FDS activations: all components of FDS have same activation
            x=activations(1,7);            
        elseif i==11||i==12||i==13||i==14 % Set FDP activations: all components of FDP have same activation
            x=activations(1,8); 
        elseif i==15||i==16||i==17||i==18 % Set EDC activations: all components of EDC have same activation
            x=activations(1,9);
        elseif i>24 % Set activation for intrinsic muscles to 1
            x=1;
        else % Set activation of remaining muscles
            x=activations(1,i-9); 
        end
            
       control_column = ArrayDouble(); 
       controls.getDataColumn(i-1,control_column); % Get the previous activation from the control file
              
       muscle=MS.get(i-1); % muscle list counts from 0:37 not 1:38
       muscleMillard=Millard2012EquilibriumMuscle.safeDownCast(muscle);
       muscleMillard.setDefaultActivation(x);% Set the default activation in the model to match the current activation
        for j=0:1
            control_column.set(j,x); % Set the values in the columns of the control file to current activation     
        end
        controls.setDataColumn(i-1,control_column); % Set data column for the specific muscle
        clear control_column; clear x;
    end
    
    % print controls file and new model
    controls.print('controls.sto');
    myModel.print('default_activation.osim');
    
    % Run forward dynamics
    FT=ForwardTool('setup.xml');
    FT.run();
    
    % Calculate Grip Force
    % Update storage file to where output is specified use full path
    JRF=Storage('C:\Users\dmcfarland\Desktop\Grip\grip_strength\grip_strength_JointReaction_ReactionLoads.sto'); %Load output of simulation
    Y = ArrayDouble(); % Variable to store Y and Z component of Grip force 
    Z = ArrayDouble(); % these components are in the transverse plane of the hand
    JRF.getDataColumn(1,Y); % get Y data
    JRF.getDataColumn(2,Z); % get Z data
    JRF_SIZE=Y.getSize(); % get # of data points
    for i=0:(Y.getSize()-1) % Loop through the data set and calculate force along the axis of dynamometer
        F_Vec=[Y.get(i);-Z.get(i)]; % load components of force in vector 
        transformation=[cosd(20) -sind(20); sind(20) cosd(20)]; % 20 degree transformation
        F_vec_along_dyna=transformation*F_Vec; %transfrom
        F_vec_y(i+1)=F_vec_along_dyna(1); %force along dynamometer
    end
    f=-mean(F_vec_y); % calculate the average grip force during the simulation

    % Load states file with information on wrist posture
    STATES=Storage('C:\Users\dmcfarland\Desktop\Grip\grip_strength\grip_strength_states_degrees.mot');
    labels=STATES.getColumnLabels(); % get column labels
    deviation = ArrayDouble(); % Variable to store deviation and flexion posture 
    flexion = ArrayDouble();
    STATES.getDataColumn(labels.getitem(1),deviation); % get deviation posture during the simulation
    STATES.getDataColumn(labels.getitem(3),flexion);   % get flexion posture during the simulation
    states_size=deviation.getSize(); % get number of data points 
    for i=1:states_size % loop through data points 
        dev=deviation.get(i-1)*180/pi; % get deviation value at each time instance
        flex=flexion.get(i-1)*180/pi; % get flexion value at each time instance
        penalty1(i)=abs(flex+35); % calculate difference from starting posture for flexion
        penalty2(i)=abs(dev-7); % calculate difference from starting posture for deviation
    end      
    PEN1=20*mean(penalty1);  % create a penalty term for flexion
    PEN2=20*mean(penalty2);  % create a penalty term for deviation
 
    f=f+PEN1+PEN2; % add penalty terms to objective function
end