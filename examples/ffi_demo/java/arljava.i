%module arljava

%pragma(java) jniclasscode=%{
   static {
     System.loadLibrary("arljava");
   }
%}

%{
#include "../src/arlwrap.h"
#include "../src/wrap_support.h"   
%}

%include "carrays.i"   

%array_class(int, intArray);

%include "../src/arlwrap.h"
%include "../src/wrap_support.h"
