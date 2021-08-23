SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
gfortran -fdec-structure $SCRIPT_DIR/Drivers/Dp/driver.f $SCRIPT_DIR/Src/Dp/src.f -o dscpack
printf '1\n8\n1\n3\n2\n' | ./dscpack
python3 plot_unf.py
