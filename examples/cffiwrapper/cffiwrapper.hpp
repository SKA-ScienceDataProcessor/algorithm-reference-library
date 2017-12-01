#ifndef CFFIWRAPPER_HPP
#define CFFIWRAPPER_HPP

#include <python3.6m/Python.h>
#include <string>
#include <iostream>

namespace ska{

  class module_exception: public std::exception{
    virtual const char* what() const throw(){
      return "Unable to load module";
    }
  };

  inline PyObject* convert_to_python(const double& value){
    return PyFloat_FromDouble(value);
  }
  
  inline  PyObject* convert_to_python(const long& value){
    return PyLong_FromLong(value);
  }
  
  template<typename T> T convert_from_python(PyObject* pyobject){}
    
  template<> long convert_from_python<long>(PyObject* pyobject){
    return PyLong_AsLong(pyobject);
  }
  
  template<> int convert_from_python<int>(PyObject* pyobject){
    return (int) convert_from_python<long>(pyobject);
  }
  
  template<> double convert_from_python<double>(PyObject* pyobject){
    return PyFloat_AsDouble(pyobject);
  }
  template<> float convert_from_python<float>(PyObject* pyobject){
    return (float) convert_from_python<double>(pyobject);
  }
  template<> void convert_from_python<void>(PyObject* pyobject){
   
  }

  template<typename return_value_type, typename... argument_types> class cffi_wrapper{
  private:
    PyObject *module, *function;
    int i_argument;
    PyObject *python_arguments;
          
    inline void set_argument(int i_argument, double value){
      PyTuple_SetItem(python_arguments, i_argument, convert_to_python(value));
    }
    inline void set_argument(int i_argument, float value){
      set_argument(i_argument, (double) value);
    }
    inline void set_argument(int i_argument, long value){
      PyTuple_SetItem(python_arguments, i_argument, convert_to_python(value));
    }
    void set_argument(int i_argument, int value){
      set_argument(i_argument, (long) value);
    }
   
    template <typename rv_T, typename...> void set_arguments(){
    }
    template <typename rv_T, typename type> void set_arguments(type value){
      set_argument(i_argument, value);
      ++i_argument;
    }
    template<typename rv_T, typename first_type, typename... remaining_types> void set_arguments(first_type first_argument,  remaining_types ... remaining_arguments){
      set_arguments<first_type>(first_argument);
      set_arguments<remaining_types...>(remaining_arguments...);
    }

  public:
    cffi_wrapper(const std::string& module_name, const std::string& function_name){
      Py_Initialize();   // perhaps move at beginning of main
      module=PyImport_Import(PyUnicode_DecodeFSDefault(module_name.c_str()));
      if(module==NULL){
        throw module_exception();
      }
      function=PyObject_GetAttrString(module, function_name.c_str());
      if(function==NULL){
        throw module_exception();
      }
      if(!PyCallable_Check(function)){
        throw module_exception();  
      }
      python_arguments=PyTuple_New(sizeof...(argument_types));
    }
    ~cffi_wrapper(){
      // Py_DECREF(module);
      // Py_DECREF(function);
      // Py_DECREF(python_arguments);
      Py_Finalize();  // perhaps move at end of main
    }
    return_value_type run(argument_types ... arguments){
      i_argument=0;
      set_arguments<return_value_type, argument_types...>(arguments...);
      return convert_from_python<return_value_type>(PyObject_CallObject(function, python_arguments));
    }
  };

}


#endif
