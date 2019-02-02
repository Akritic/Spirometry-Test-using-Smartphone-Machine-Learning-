function [y,b,a] = apply_filter(t,x,fc)

fs=1/(t(2)-t(1));
[b,a]=butter(5,fc/(fs/2));
y=filter(b,a,x);

end
%------------------------------------------------------------------------%