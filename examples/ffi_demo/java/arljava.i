%module arljava

%pragma(java) jniclasscode=%{
   static {
     System.loadLibrary("arljava");
   }
%}

%{
#include "../include/arlwrap.h"
#include "../include/wrap_support.h"   
%}

%include "carrays.i"   

%array_class(int, intArray);
%array_class(double, doubleArray);

%include "../include/arlwrap.h"
%include "../include/wrap_support.h"
