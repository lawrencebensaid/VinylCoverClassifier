
% Enum to keep track of Certainty of the image match
% < 6 matches is Low
% > 5 && < 16 matches is Medium
% > 15 matches is High
classdef Certainty
   enumeration
      Low, Medium, High
   end
    methods
        function certainty = determineCertainty(points)
         if(points > 15)
             certainty = Certainty.High;
        elseif (points > 5)
                certainty = Certainty.Medium;
         else
             certainty = Certainty.low;
         end
      end
   end
end