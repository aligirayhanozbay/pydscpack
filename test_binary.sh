gfortran Drivers/Dp/driver.f Src/Dp/src.f -o dscpack
printf '1\n8\n1\n3\n2\n' | ./dscpack
python3 plot_unf.py
