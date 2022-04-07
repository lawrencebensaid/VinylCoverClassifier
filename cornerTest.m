clear;

I = imread("album2.png");

I = rgb2gray(I);
I = double(I);

filtered_image = zeros(size(I));
Mx = [-1 0 1; -2 0 2; -1 0 1];
My = [-1 -2 -1; 0 0 0; 1 2 1];
for i = 1:size(I, 1) - 2
    for j = 1:size(I, 2) - 2
        Gx = sum(sum(Mx.*I(i:i+2, j:j+2)));
        Gy = sum(sum(My.*I(i:i+2, j:j+2)));
        
        filtered_image(i+1, j+1) = sqrt(Gx.^2 + Gy.^2);
    end
end

filtered_image = uint8(filtered_image);
%figure, imshow(filtered_image); title('Filtered Image');
  
thresholdValue = 200;
output_image = max(filtered_image, thresholdValue);
output_image(output_image == round(thresholdValue)) = 0;
  
% Displaying Output Image
output_image = im2bw(output_image, graythresh(output_image));
%figure, imshow(output_image); title('Edge Detected Image');

SE = strel('disk', 16);

bw = imclose(output_image, SE);

bw = imfill(bw,'holes');

I = mat2gray(I);

points = detectHarrisFeatures(bw);
figure; imshow(I); hold on;
plot(points.selectStrongest(50));

x = points.Location;
x