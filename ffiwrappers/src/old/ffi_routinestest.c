#include <stdio.h>


static (*print_hello_from_python)(char *name);

int main(){

  print_hello_from_python("Montse");
  return 0;

}
