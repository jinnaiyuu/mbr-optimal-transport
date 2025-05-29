# Default parameters are set to run a debug experiment.

DOMAIN=samsum
MODEL=bart-large-cnn-samsum
NLINES=3
NSAMPLES=5
TEMPERATURE=1.0
EPS=0.01
TOPK=0
TOPP=1.0
SIM=bertscore
EVAL=sacrebleu
ALGORITHM=None
DEBUG=0
RECOMPUTE=""

RZERO=4
PALPHA=0.9
DOSAMPLE=1
DIVERSITY=1.0
DIVERSEK=4
PAIRWISE=sacrebleu

BUDGETS=-1

STARTITER=0
CACHEDIR=~/. # TODO: Set by option

while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h:c:f: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    p)
        PROMPT=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    f)
        TEMPERATURE=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        # Nucleus sampling
        TOPP=${OPTARG};;
    i)
        # TODO: Long options
        SIM=${OPTARG};;
    v)
        EVAL=${OPTARG};;
    a)
        # TODO: Enable arguments for algorithm e.g. k, div_pen
        ALGORITHM=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute_matrix";;
    u)
        # APPROXIMATION
        BUDGETS=${OPTARG};;
    o)
        # APPROXIMATION
        RZERO=${OPTARG};;
    h)
        # APPROXIMATION
        PALPHA=${OPTARG};;
    t)
        # DIVERSITY
        DOSAMPLE=0
        DIVERSITY=${OPTARG};;
    z)
        # DIVERSITY
        DIVERSEK=${OPTARG};;
    w)
        # DIVERSITY
        PAIRWISE=${OPTARG};;
    c)
        STARTITER=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

if [ "$ALGORITHM" == "beam" ]; then
    DOSAMPLE=-1
    DIVERSITY=0
elif [ "$ALGORITHM" == "dbs" ]; then
    DOSAMPLE=0
fi

# TODO: Refactor this with option
if [ "$DOMAIN" == "xsum" ]; then
    DATADIR=xsum
elif [ "$DOMAIN" == "cnndm" ]; then
    DATADIR=cnn_cln
elif [ "$DOMAIN" == "samsum" ]; then
    DATADIR=None
elif [[ $DOMAIN == "wmt21"* ]]; then
    DATADIR=wmt21
elif [[ $DOMAIN == "wmt19"* ]]; then
    DATADIR=wmt19-text
# elif [ "$DOMAIN" == "wmt19.en-ru" ]; then
#     DATADIR=wmt19-text
# elif [ "$DOMAIN" == "wmt19.de-en" ]; then
#     DATADIR=wmt19-text
# elif [ "$DOMAIN" == "wmt19.ru-en" ]; then
#     DATADIR=wmt19-text
elif [ "$DOMAIN" == "iwslt17.fr-en" ]; then
    DATADIR=iwslt17-text
elif [ "$DOMAIN" == "nocaps" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "mscoco" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "mscoco-ft" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "e2e_nlg" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "common_gen" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "squad_v2" ]; then
    DATADIR=None
else
    echo "No cache available for $DOMAIN. Loading from huggingface datasets."
    DATADIR=None
    # exit 1
fi

# TODO: Check parameters here

# if [ "$DEBUG" == "1" ]; then
#     echo "Debug mode: using local files to run mbr"
#     echo "Make sure that the sample and matrix files are placed at $CACHEDIR/samples/$DOMAIN/$MODEL"
# else 
#     # Sampling files
#     SAMPLES3DIR="mbr/samples/$DOMAIN/$MODEL"

#     PATSTRING=`python3 experiments/get_sample_names.py $EPS $TOPK $TOPP $DOSAMPLE $DIVERSEK $DIVERSITY $TEMPERATURE`
#     echo "downloading $SAMPLES3DIR/ with pattern $PATSTRING"
#     mkdir -p $CACHEDIR/fairseq/$SAMPLES3DIR
#     python3 experiments/download.py $SAMPLES3DIR/ --dist $CACHEDIR --pattern $PATSTRING
#     echo "done downloading samples"
#     # mkdir -p $CACHEDIR/samples/$DOMAIN/$MODEL
#     # mv ./fairseq/$SAMPLES3DIR/* ./samples/$DOMAIN/$MODEL/
#     # find $CACHEDIR/fairseq/$SAMPLES3DIR/ -type f  -exec mv {} $CACHEDIR/samples/$DOMAIN/$MODEL/ \;

#     # dataset files
#     if [ "$DATADIR" == "None" ]; then
#         echo "uses huggingface dataset"
#     else
#         echo "downloading $DATADIR"
#         python3 experiments/download.py $DATADIR
#         mkdir -p ./dataset/$DATADIR
#         mv ./fairseq/$DATADIR/* ./dataset/$DATADIR/
#     fi

#     # Matrix files
#     if [ "$RECOMPUTE" == "" ] && [ "$DOSAMPLE" == "1" ]; then
#         MATRIX3DIR="mbr/matrix/$DOMAIN/$MODEL"
#         echo "downloading $MATRIX3DIR/ with pattern ${PATSTRING}_*"
#         mkdir -p $CACHEDIR/fairseq/$MATRIX3DIR
#         python3 experiments/download.py $MATRIX3DIR/ --dist $CACHEDIR --pattern "${PATSTRING}_*"
#         # mkdir -p $CACHEDIR/matrix/$DOMAIN/$MODEL
#         # mv ./fairseq/$MATRIX3DIR/* ./matrix/$DOMAIN/$MODEL/
#         # find ./fairseq/$MATRIX3DIR/ -type f  -exec mv {} $CACHEDIR/matrix/$DOMAIN/$MODEL/ \;
#         # find $CACHEDIR/fairseq/$MATRIX3DIR/ -type f  -exec mv {} $CACHEDIR/matrix/$DOMAIN/$MODEL/ \;
#         echo "done downloading matrices"
#     fi

# fi

# Return an error if the python script fails
set -e
SAMPLES3DIR="samples/$DOMAIN/$MODEL"
MATRIX3DIR="matrix/$DOMAIN/$MODEL"


python3 mbr/mbr_engine.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./$SAMPLES3DIR \
    --matrix_dir ./$MATRIX3DIR \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --temperature $TEMPERATURE \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --sim $SIM \
    --eval $EVAL \
    --algorithm $ALGORITHM \
    $RECOMPUTE \
    --do_sample $DOSAMPLE --diversity_penalty $DIVERSITY \
    --diverse_k $DIVERSEK \
    --approx_budgets $BUDGETS \
    --pairwise_eval $PAIRWISE \
    --r_0 $RZERO --pruning_alpha $PALPHA

if [ "$DEBUG" == "1" ]; then
    echo "done!"
    # "TODO: print results"
    #  "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(dataset, model_n, n_samples, 
    #                                                                 epsilon, topk, topp, sim, eval_func, postfix)

    # cat ./results/$DOMAIN/$MODEL/0000_eps-0.02_topk-0_topp-1.0_sim-bertscore_eval-sacrebleu.txt
fi
