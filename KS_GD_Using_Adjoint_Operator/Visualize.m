clear all
data_file = 'L2_Residue_E3.h5';
L2 = h5read(data_file,'/L2_Residue');

x = linspace(1,40000,40000);
plot(x(1,200:20000),L2(1,200:20000))