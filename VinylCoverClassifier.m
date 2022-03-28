% Import
I = imread("data/test/2260.jpg");
I = rgb2gray(I);

% Process
Iedges = edge(I, "canny", .2);

% Display
figure;
imshow(Iedges);
