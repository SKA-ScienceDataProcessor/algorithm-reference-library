#include "cffiwrapper.hpp"
#include <iostream>

int main(){

  // double f(int a, int b, double c)=(a+b)*c;
  ska::cffi_wrapper<double, int, int, double> f("module", "f");
  std::cout << f.run(1, 2, 0.5) << "\n";
  
  return 0;
}
