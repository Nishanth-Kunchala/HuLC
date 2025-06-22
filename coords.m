function [x, y, z] = coords()
    u = udpport("LocalPort", 5005);
    disp('Waiting for valid UDP coordinates...');

    x = NaN;
    y = NaN;

    while true
        if u.NumBytesAvailable > 0
            data = readline(u);
            try
                coords = jsondecode(char(data));
                x = coords(1);
                y = coords(2);
                z = coords(3);
                break;
            catch
                warning("Bad data received.");
            end
        end
        pause(0.05);
    end

    clear u;
end
