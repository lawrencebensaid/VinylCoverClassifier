function bestMatch = determineBestMatch(imageNames,targetImageName)
%determineBestMatch Returns the picture with the most matching points
%   Detailed explanation goes here
bestMatch = targetImageName;

highestMatches = 0;
bestMatch;

    for k = 1:length(imageNames)
        matchingpoints = match(imageNames(k), targetImageName);
    
        if(matchingpoints > highestMatches)
            highestMatches = matchingpoints;
            bestMatch = imageNames(k);
        end
    end
end