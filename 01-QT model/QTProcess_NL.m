function dHdt = QTProcess_NL(t,x,param,v_1,v_2)
% Function that contains the differential equations for the non-linear
% Quadruple Tank benchmark process.
% x(1) = h1
% x(2) = h2
% x(3) = h3
% x(4) = h4
dh1dt = ( -1*param.a1*sqrt(2*param.g*x(1)) + param.a3*sqrt(2*param.g*x(3)) + param.gamma1(t)*param.k1*v_1(t) )/param.A1;
dh2dt = ( -1*param.a2*sqrt(2*param.g*x(2)) + param.a4*sqrt(2*param.g*x(4)) + param.gamma2(t)*param.k2*v_2(t) )/param.A2;
dh3dt = ( -1*param.a3*sqrt(2*param.g*x(3)) + (1-param.gamma2(t))*param.k2*v_2(t) )/param.A3;
dh4dt = ( -1*param.a4*sqrt(2*param.g*x(4)) + (1-param.gamma1(t))*param.k1*v_1(t) )/param.A4;

dHdt = [dh1dt,dh2dt,dh3dt,dh4dt]';

end