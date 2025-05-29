set -e

if [[ -z ${JOB_NAME} ]]; then
    JOB_NAME="docmt-evaluate-$(date +%Y%m%d-%H%M%S)"         # Job Name
fi

python3 -m mt_metrics_eval.mtme --download

python3 score_doc-metrics.py "$@" | tee ${JOB_NAME}.txt

MESSAGE=`cat ${JOB_NAME}.txt`
echo '{"text":"'"${MESSAGE}"'"}'

curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"'"${MESSAGE}"'"}' https://hooks.slack.com/services/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