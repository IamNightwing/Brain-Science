
img = ones(28,28,3) * 255;    % 3 channels: R G B
img = uint8(img);               % we are using integers 0 .. 255

for a = 0: 10: 255
    for b = 0: 10: 255
        for c = 0: 10: 255
            s0 = 'F:\Brain Science\Dataset\'; s1 = int2str(a);s2 = int2str(b);s3= int2str(c); s4 = '.png';
            filename = strcat(s0,s1,s2,s3,s4);
            img(:, :, 1) = a;       % set R component to maximum value
            img(:, :, 2) = b;
            img(:, :, 3) = c;% clear G and B components
            imwrite(img, filename);
        end
    end    
end



%imshow(img);