function img = getPictureFromWebcam()
%GETPICTUREFROMWEBCAM Creates snapshot of webcam based on OS
%   Detailed explanation goes here

type = '';
if ismac
   type = 'macvideo';
elseif isunix
    type = 'linuxvideo';
elseif ispc
    type = 'winvideo';
else
    disp('Platform not supported');
end

    x = videoinput(type, 1);
    img = getsnapshot(x);

end

