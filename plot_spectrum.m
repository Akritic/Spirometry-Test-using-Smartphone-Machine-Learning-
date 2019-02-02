function plot_spectrum(time, signal)

complex_spectrum = fftshift(fft(signal));
Ts = time(1)-time(2);                     
Fs = 1/Ts;
N = length(signal);
dF = Fs/length(signal);
f = -Fs/2:dF:Fs/2-dF;           % hertz
plot(f,abs(complex_spectrum)/N);
    
end

%------------------------------------------------------------------------%