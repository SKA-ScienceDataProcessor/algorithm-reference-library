#!/usr/bin/env bash

set -o errexit -o pipefail

source "${HOME}/.bash_profile"
#source activate dask-distributed

echo "Complete environment:"
printenv

if [ \( -n "${MARATHON_APP_ID-}" \) -a \( -n "${HOST-}" \) \
    -a \( -n "${PORT_SCHEDULER-}" \) -a \( -n "${PORT_BOKEH-}" \) ]
then
    BOKEH_WHITELIST="*"
    BOKEH_APP_PREFIX=""

    if [ -n "${MARATHON_APP_LABEL_HAPROXY_1_VHOST-}" ]
    then
        BOKEH_WHITELIST="${MARATHON_APP_LABEL_HAPROXY_1_VHOST}"
    fi

    if [ -n "${MARATHON_APP_LABEL_HAPROXY_1_PATH-}" ]
    then
        BOKEH_APP_PREFIX="${MARATHON_APP_LABEL_HAPROXY_1_PATH}"
    fi

    echo ""
    echo "Command to run: "
    echo dask-scheduler --host "${HOST}" --port "${PORT_SCHEDULER}" --bokeh-port "${PORT_BOKEH}" --bokeh-whitelist "${BOKEH_WHITELIST}" --bokeh-prefix "${BOKEH_APP_PREFIX}" --use-xheaders "True" --scheduler-file "dask-scheduler-connection" --local-directory "${MESOS_SANDBOX}"

    dask-scheduler \
        --host "${HOST}" \
        --port "${PORT_SCHEDULER}" \
        --bokeh-port "${PORT_BOKEH}" \
        --bokeh-whitelist "${BOKEH_WHITELIST}" \
        --bokeh-prefix "${BOKEH_APP_PREFIX}" \
        --use-xheaders "True" \
        --scheduler-file "dask-scheduler-connection" \
        --local-directory "${MESOS_SANDBOX}"
        # --tls-ca-file "${MESOS_SANDBOX}/.ssl/ca-bundle.crt" \
        # --tls-cert "${MESOS_SANDBOX}/.ssl/scheduler.crt" \
        # --tls-key "${MESOS_SANDBOX}/.ssl/scheduler.key" \
else
    if [ \( -n "${KUBERNETES_SERVICE_HOST-}" \) ]
    then

        echo ""
        echo "Command to run: "
        echo dask-scheduler --host "${DASK_HOST_NAME}" --port "${DASK_PORT_SCHEDULER}" --bokeh-port "${DASK_PORT_BOKEH}" --bokeh-whitelist "${DASK_BOKEH_WHITELIST}" --bokeh-prefix "${DASK_BOKEH_APP_PREFIX}" --use-xheaders "True" --scheduler-file "dask-scheduler-connection" --local-directory "${DASK_LOCAL_DIRECTORY}"

        dask-scheduler \
            --host "${DASK_HOST_NAME}" \
            --port "${DASK_PORT_SCHEDULER}" \
            --bokeh-port "${DASK_PORT_BOKEH}" \
            --bokeh-whitelist "${DASK_BOKEH_WHITELIST}" \
            --bokeh-prefix "${DASK_BOKEH_APP_PREFIX}" \
            --use-xheaders "True" \
            --scheduler-file "dask-scheduler-connection" \
            --local-directory "${DASK_LOCAL_DIRECTORY}"
    else
        dask-scheduler "$@"
    fi
fi
