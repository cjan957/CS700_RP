clc;
clear;

I1 = imread('I1_000163.png');
I2 = imread('I2_000163.png');

I3 = imread('disparity.png');

I1_res = imread('I1_000163_res.png');
I2_res = imread('I2_000163_res.png');

I3_res = imread('disparity_res.png');


figure
imshow(I3);
title('Disparity From C');

figure
imshow(I3_res);
title('Disparity From C (resized)');

disparityRange = [0,256];
disparityMap = disparity(I1,I2,'DisparityRange',disparityRange, 'BlockSize',17, 'Method','BlockMatching');

disparityMap_res = disparity(I1_res,I2_res,'DisparityRange',disparityRange, 'BlockSize',9, 'Method','BlockMatching');

val = disparityMap(233, 690);

val_res = disparityMap_res(117,345);

focalLength = 647.1884; % In Pixels
baseline = -(-3.745166) / 6.471884; % Distance between the two cameras

distance = (focalLength * baseline) / val;

distance_res = (focalLength * baseline) / val_res;


disp("Pixel Value is: " + num2str(val));
disp("Distance is : " + num2str(distance) + " m");

disp("Pixel Value (resized) is: " + num2str(val_res));
disp("Distance is (resized): " + num2str(distance_res) + " m");

figure 
imshow(disparityMap,disparityRange);
title('Disparity Map');

figure 
imshow(disparityMap_res,disparityRange);
title('Disparity Map (resized)');


