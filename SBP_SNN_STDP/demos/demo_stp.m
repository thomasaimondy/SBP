t = zeros(1000,1);
tsp = 200;
spike = delta(t,tsp);
U = 0.45;
taof = 200;%ms
taod = 2500;%ms

u = 0;
x = 1;
a = zeros(length(spike),1);
b = zeros(length(spike),1);

for t = 1:length(spike)
    u = u - u/taof + U * spike(t) * (1-u);
    a(t) = u;
    x = x + (1-x) / taod - u * x * spike(t);
    b(t) = x;
end
plot(a);
legend('u')
hold on 
plot(b,'r');
hold on
plot(spike*0.05,'g')
legend('u','x','spike')
    