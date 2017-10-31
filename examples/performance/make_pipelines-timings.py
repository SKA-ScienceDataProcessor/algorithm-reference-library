
import sys

if len(sys.argv) == 1:
    print("make_pipelines-timings.sh [HOSTNAME NUMBER_NODES NUMBER_PROCS_PER_NODE "
          "NUMBER_FREQUENCY_WINDOWS NUMBER_THREADS EXECUTION_TIME ]")
    
if len(sys.argv) > 1:
    HOSTNAME = sys.argv[1]
else:
    HOSTNAME ='darwin'

if len(sys.argv) > 2:
    NUMBER_NODES=int(sys.argv[2])
else:
    NUMBER_NODES=1
    
if len(sys.argv)>3:
    NUMBER_PROCS_PER_NODE=int(sys.argv[3])
else:
    NUMBER_PROCS_PER_NODE=4

NUMBER_TASKS = NUMBER_PROCS_PER_NODE * NUMBER_NODES

if len(sys.argv)>4:
    NUMBER_FREQUENCY_WINDOWS=int(sys.argv[4])
else:
    NUMBER_FREQUENCY_WINDOWS = 1

if len(sys.argv)>5:
    NUMBER_THREADS=int(sys.argv[5])
else:
    NUMBER_THREADS = 1

if len(sys.argv)>6:
    EXECUTION_TIME=sys.argv[6]
else:
    EXECUTION_TIME='12:00:00'

template_file = 'submit_%s_template' % HOSTNAME
outfile = \
    'submit_HOSTNAME_NUMBER_NODESnodes_NUMBER_PROCS_PER_NODEprocspernode_NUMBER_FREQUENCY_WINDOWSnfreqwin_NUMBER_THREADSthreads'

def sub(s):
    return s.replace('HOSTNAME', str(HOSTNAME)).replace(
        'NUMBER_NODES', str(NUMBER_NODES)).replace(
        'NUMBER_TASKS', str(NUMBER_TASKS)).replace(
        'NUMBER_PROCS_PER_NODE', str(NUMBER_PROCS_PER_NODE)).replace(
        'NUMBER_FREQUENCY_WINDOWS', str(NUMBER_FREQUENCY_WINDOWS)).replace(
        'NUMBER_THREADS', str(NUMBER_THREADS)).replace(
        'EXECUTION_TIME', EXECUTION_TIME)

outfile = sub(outfile)

with open(template_file, 'r') as template:
    script = template.readlines()
    
with open(outfile, 'w') as edited:
    for line in script:
        edited.write(sub(line))



