#include <Python.h>
#include <stdio.h>

/* Simple exit-on-error */
void pycheck(PyObject *obj)
{
	if (!obj) {
		PyErr_Print();
		exit(1);
	}
}

/* In: module name, function name
 * Out: function address */
size_t get_ffi_fn_addr(const char* module, const char* fn_name)
{
	PyObject *mod, *fn, *fn_addr;

	pycheck(mod = PyImport_ImportModule(module));
	pycheck(fn = PyObject_GetAttrString(mod, fn_name));
	pycheck(fn_addr = PyObject_GetAttrString(fn, "address"));

	return PyNumber_AsSsize_t(fn_addr, NULL);
}

int main(int argc, char **argv)
{
	int i, count;
	Py_Initialize();

	int (*pfun)(char *) = get_ffi_fn_addr("ffi_pyroutines", "ffi_phfp");
	int len = pfun("C-based interface user");
	printf("Name length was: %d\n", len);

	int (*cfun)(void) = get_ffi_fn_addr("ffi_pyroutines", "ffi_gcfp");
	for(i = 0; i < 10; i++) {
		count = cfun();
		printf("\t%d\n", count);
	}

	return 0;
}
