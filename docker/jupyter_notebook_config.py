import errno
import os
import stat
import subprocess


from jupyter_core.paths import jupyter_data_dir
from notebook.auth import passwd

# Setup the Notebook to listen on all interfaces on port 8888 by default
c.NotebookApp.ip = '*'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# Configure Networking while running under Marathon:
if 'MARATHON_APP_ID' in os.environ:
    if 'PORT_JUPYTER' in os.environ:
        c.NotebookApp.port = int(os.environ['PORT_JUPYTER'])

    # Set the Access-Control-Allow-Origin header
    c.NotebookApp.allow_origin = '*'

    # Set Jupyter Notebook Server password to 'jupyter-<Marathon-App-Prefix>'
    # e.g., Marathon App ID '/foo/bar/app' maps to password: 'jupyter-foo-bar'
    MARATHON_APP_PREFIX = \
        '-'.join(os.environ['MARATHON_APP_ID'].split('/')[:-1])
    c.NotebookApp.password = passwd('jupyter{}'.format(MARATHON_APP_PREFIX))

    # Allow CORS and TLS from behind Marathon-LB/HAProxy
    # Trust X-Scheme/X-Forwarded-Proto and X-Real-Ip/X-Forwarded-For
    # Necessary if the proxy handles SSL
    if 'MARATHON_APP_LABEL_HAPROXY_GROUP' in os.environ:
        c.NotebookApp.trust_xheaders = True

    if 'MARATHON_APP_LABEL_HAPROXY_0_VHOST' in os.environ:
        c.NotebookApp.allow_origin = \
            'http://{}'.format(
                os.environ['MARATHON_APP_LABEL_HAPROXY_0_VHOST']
            )

    if 'MARATHON_APP_LABEL_HAPROXY_0_REDIRECT_TO_HTTPS' in os.environ:
        c.NotebookApp.allow_origin = \
            'https://{}'.format(
                os.environ['MARATHON_APP_LABEL_HAPROXY_0_VHOST']
            )

    # Set the Jupyter Notebook server base URL to the HAPROXY_PATH specified
    if 'MARATHON_APP_LABEL_HAPROXY_0_PATH' in os.environ:
        c.NotebookApp.base_url = \
            os.environ['MARATHON_APP_LABEL_HAPROXY_0_PATH']

    # Setup TLS
    if 'USE_HTTPS' in os.environ:
        SCHEDULER_TLS_CERT = os.environ.get('TLS_CERT_PATH', '/'.join(
            [os.environ['MESOS_SANDBOX'],
             '.ssl',
             'scheduler.crt']))
        c.NotebookApp.certfile = SCHEDULER_TLS_CERT
        SCHEDULER_TLS_KEY = os.environ.get('TLS_KEY_PATH', '/'.join(
            [os.environ['MESOS_SANDBOX'],
             '.ssl',
             'scheduler.key']))
        c.NotebookApp.keyfile = SCHEDULER_TLS_KEY

# Set a certificate if USE_HTTPS is set to any value
PEM_FILE = os.path.join(jupyter_data_dir(), 'notebook.pem')
if 'USE_HTTPS' in os.environ:
    if not os.path.isfile(PEM_FILE):
        # Ensure PEM_FILE directory exists
        DIR_NAME = os.path.dirname(PEM_FILE)
        try:
            os.makedirs(DIR_NAME)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(DIR_NAME):
                pass
            else:
                raise
        # Generate a certificate if one doesn't exist on disk
        subprocess.check_call(['openssl', 'req', '-new', '-newkey', 'rsa:2048',
                               '-days', '365', '-nodes', '-x509', '-subj',
                               '/C=XX/ST=XX/L=XX/O=generated/CN=generated',
                               '-keyout', PEM_FILE, '-out', PEM_FILE])
        # Restrict access to PEM_FILE
        os.chmod(PEM_FILE, stat.S_IRUSR | stat.S_IWUSR)
    c.NotebookApp.certfile = PEM_FILE

# Set a password if JUPYTER_PASSWORD is set
if 'JUPYTER_PASSWORD' in os.environ:
    c.NotebookApp.password = passwd(os.environ['JUPYTER_PASSWORD'])
    del os.environ['JUPYTER_PASSWORD']
