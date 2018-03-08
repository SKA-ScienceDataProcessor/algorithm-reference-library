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

%include "../include/arlwrap.h"
%include "../include/wrap_support.h"
