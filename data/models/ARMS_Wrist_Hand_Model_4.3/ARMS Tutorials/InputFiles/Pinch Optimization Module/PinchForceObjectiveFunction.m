function f = PinchForceObjectiveFunction(activations)
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
	
%PINCHFORCEOBJECTIVEFUNCTION: This is an objective function to optimize pinch
%force from a computational OpenSim hand model. This function takes in
%activations for the individual muscles, updates the control file for a
%forward dynamic simulation, and then runs the forward dynamic simulation
%via a predefined setup file. The objective value is Pinch force which is
%defined as the magnitude of the constraint force between a massless body and the
%distal thumbtip

	% Import OpenSim modeling classes
	import org.opensim.modeling.*
    
    %Get model and muscle set
    myModel=Model('Pinch_Model.osim');% Get model
    MS=myModel.getMuscles(); % Get muscle set
    MS_size=MS.getSize(); % Get # of muscles 
    
    % Update controls file and model with current activations    
    controls = Storage('Controls.sto'); % Load the controls file.   
    for i=1:14 % pinch simulations have 14 muscles
        x=activations(1,i); % set activation       
       control_column = ArrayDouble(); 
       controls.getDataColumn(i-1,control_column);  % Get the previous activation from the control file      
       muscle=MS.get(i-1); % muscle list counts from 0:13 not 1:14
       muscleMillard=Millard2012EquilibriumMuscle.safeDownCast(muscle);
       muscleMillard.setDefaultActivation(x); % Set the default activation in the model to match the current activation
        for j=0:1
            control_column.set(j,x); % Set the values in the columns of the control file to current activation                  
        end
        controls.setDataColumn(i-1,control_column); % Set data column for the specific muscle
        clear control_column; clear x;
    end
    
    % print controls file and new model
    controls.print('Controls.sto');
    myModel.print('default_activation.osim');
    
    % Run forward dynamics
    FT=ForwardTool('setup.xml');
    FT.run();
    
    % Calculate Pinch Force
	% Update storage file to where output is specified use full path
    FR=Storage('C:\Users\Dcmcfarl\Desktop\ARMS_Wrist_Hand_Model\ARMS Tutorials\4.3\InputFiles\Pinch Optimization Module\pinch_force\pinch_force_ForceReporter_forces.sto');
    X = ArrayDouble();% Variable to store X, Y, and Z component of Pinch force 
    Y = ArrayDouble();
    Z = ArrayDouble();
    FR.getDataColumn(78,X); % get X data
    FR.getDataColumn(79,Y); % get Y data
    FR.getDataColumn(80,Z); % get Z data
    FR_SIZE=Y.getSize(); % get # of data points
    for i=0:(Y.getSize()-1) % Loop through the data set and store in a vector
       force_x(i+1)=X.get(i); % x force vector
       force_y(i+1)=Y.get(i); % y force vector; y force is the palmar pinch force
       force_z(i+1)=Z.get(i); % z force vector
    end
    
    if abs(mean(force_x))>=abs((0.17*mean(force_y))) % add a penalty term if force in x direction is > 17% of the palmar force
        p1=10;
    else
        p1=0;
    end
    if abs(mean(force_z))>=abs((0.17*mean(force_y))) % add a penalty term if force in z direction is > 17% of the palmar force
        p2=10;
    else
        p2=0;
    end
    f=mean(force_y)+p1+p2; % add in the penalty terms
    
    % Load states file with information on wrist posture
    STATES=Storage('C:\Users\Dcmcfarl\Desktop\ARMS_Wrist_Hand_Model\ARMS Tutorials\4.3\InputFiles\Pinch Optimization Module\pinch_force\pinch_force_states_degrees.mot');
    labels=STATES.getColumnLabels(); % get column labels
    deviation = ArrayDouble(); % Variable to store deviation and flexion posture
    flexion = ArrayDouble();
    STATES.getDataColumn(labels.getitem(1),deviation); % get deviation posture during the simulation
    STATES.getDataColumn(labels.getitem(3),flexion);   % get flexion posture during the simulation
    states_size=deviation.getSize(); % get number of data points
    for i=1:states_size % loop through data points
        dev=deviation.get(i-1)*180/pi; % get deviation value at each time instance
        flex=flexion.get(i-1)*180/pi; % get flexion value at each time instance
        penalty1(i)=abs(flex);% calculate how far wrist moved from neutral in flexion or extension
        penalty2(i)=abs(dev); % calculate how far wrist moved from neutral in radial or ulnar
    end 
    if max(penalty1)>=5 % add in penalty if wrist moved by more than an average of 5 degrees in flexion/extension
        PEN1=20*mean(penalty1);
    else
        PEN1=0;
    end
    if max(penalty2)>=5 % add in penalty if wrist moved by more than an average of 5 degrees in radial/ulnar
        PEN2=20*mean(penalty2);
    else
        PEN2=0;
    end
         
    f=f+PEN1+PEN2; % add penalty terms to objective function
end