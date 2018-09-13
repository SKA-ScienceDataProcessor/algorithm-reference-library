"""Offers he base class for arlexecute-based unit tests"""

class ARLExecuteTestCase(object):
    """Sets up the arlexecute global object as appropriate, and closes it when done"""

    def setUp(self):
        super(ARLExecuteTestCase, self).setUp()

        import os
        from wrappers.arlexecute.execution_support.arlexecute import arlexecute
        use_dlg = os.environ.get('ARL_TESTS_USE_DLG', '0') == '1'
        use_dask = os.environ.get('ARL_TESTS_USE_DASK', '0') == '1'
        arlexecute.set_client(use_dask=use_dask, use_dlg=use_dlg)

        # Start a daliuge node manager for these tests; make sure it can see
        # the arl modules. The node manager will be shut down at tearDown
        if use_dlg:
            from dlg import tool
            arl_root = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            self.nm_proc = tool.start_process('nm', ['--dlg-path', arl_root])

    def tearDown(self):
        from wrappers.arlexecute.execution_support.arlexecute import arlexecute
        arlexecute.close()
        if arlexecute.using_dlg:
            from dlg import utils
            utils.terminate_or_kill(self.nm_proc, 10)
        super(ARLExecuteTestCase, self).tearDown()