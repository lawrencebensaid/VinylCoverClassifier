function changeImageFormat(input,output)
%CONVERTIMAGETOPGM Convert an image into desired output format
    imwrite(imread(input), output);
end