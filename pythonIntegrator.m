clc
clear

% Setup UDP port
u = udpport("LocalPort", 5005);

disp('Listening for incoming UDP data...');

while true
    if u.NumBytesAvailable > 0
        data = readline(u);
        try
            coords = jsondecode(char(data));
            disp(coords);  % Debug print

            % Call Stewart platform control
            
        catch err
            warning("Failed to parse or move");
        end
    end
    pause(0.05);  % Adjust as needed
end
