
function h=plotSurfaceQ1(X1,X2,T)
    XVEC = X1;
    YVEC = X2;
    ZVEC = T;
    size(X1)
    size(X2)
    size(T)
    F=scatteredInterpolant(XVEC,YVEC,ZVEC);
    [Xq,Yq]=meshgrid(min(XVEC):0.01:max(XVEC),min(YVEC):0.01:max(YVEC));
    Vq = F(Xq,Yq);
    h=surfc(Xq,Yq,Vq);
    xlabel('X_1');
    ylabel('X_2');
    zlabel('T');
    title('Training set');
end
