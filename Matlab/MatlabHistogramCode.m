binnumx = 100;
binnumy = 100;

load('codedFull.mat');
minx = min(codedFull(:,2));
miny = min(codedFull(:,3));
maxx = max(codedFull(:,2));
maxy = max(codedFull(:,3));
dx = (maxx - minx)/(binnumx-1);
dy = (maxy - miny)/(binnumy-1);

xrange = zeros(binnumx , 1);
yrange = zeros(binnumy , 1);
countFull = zeros(binnumx , binnumy);
xrange(1,1)=minx;
yrange(1,1)=miny;
i=1;
while(i<binnumx)
    i=i+1;
    xrange(i,1) = xrange(i-1,1) + dx;
end
i=1;
while(i<binnumy)
    i=i+1;
    yrange(i,1) = yrange(i-1,1) + dy;
end
i=0;
while(i<length(codedFull(:,1)))
    i=i+1;
    ix = floor((codedFull(i, 2) - minx + dx/2)/dx)+1;
    iy = floor((codedFull(i, 3) - miny + dy/2)/dy)+1;
    countFull(ix, iy) = countFull(ix, iy)+1;
end

mesh(xrange(:,1),yrange(:,1),countFull)
shading('interp')
surf(xrange(:,1),yrange(:,1),countFull)
colormap(flipud(parula))