load("eod.mat");  % field name: pad_leod1
waveform = pad_leod1;
clear pad_leod1;
sampling_rate = 2500000;  % sampling rate is 2.5 MHz

% remove the 0 padding of the eod
fraction_of_max = 2e-2;
max_waveform = max(abs(waveform));
ids_good = find(abs(waveform) > (max_waveform * fraction_of_max));
waveform = waveform(ids_good(1):ids_good(end));
waveform = waveform - waveform(1);
waveform = waveform / max(waveform);

% fit waveform with sum of 7 gaussians
times = ((0:(length(waveform) - 1)) / sampling_rate)';
waveform_fit = fit(times, waveform, 'gauss7');

waveform_fitted = waveform_fit(times);  % get back fitted waveform
waveform_fitted = waveform_fitted - waveform_fitted(1);
waveform_fitted = waveform_fitted / max(waveform_fitted);

% compute the supra-resolution waveform @ sampling rate 1000 times larger
times_supra1000 = ((0:(length(waveform) * 1000 - 1)) / (1000 * sampling_rate))';
waveform_fitted_supra1000 = waveform_fit(times_supra1000);
waveform_fitted_supra1000 = waveform_fitted_supra1000 - waveform_fitted_supra1000(1);
waveform_fitted_supra1000 = waveform_fitted_supra1000 / max(waveform_fitted_supra1000);

% plot the waveform and the adjusted fits
figure(1); clf();
plot(times, waveform); hold on;
plot(times, waveform_fitted);
plot(times_supra1000, waveform_fitted_supra1000);
legend()

% write to .csv
writematrix(waveform, 'eod-waveform.csv');
writematrix(waveform_fitted, 'eod-waveform-fitted.csv');
writematrix(waveform_fitted_supra1000, 'eod-waveform_fitted_supra1000.csv');

% clear Workspace
clear;clc;
