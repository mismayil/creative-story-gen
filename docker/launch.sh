#!/bin/bash

IMAGE="ic-registry.epfl.ch/nlp/ismayilz/project-c-decoding"
COMMAND=$1
CLUSTER=ic
N_GPUS=1
N_CPUS=4
JOB_PREFIX="project-csg"
JOB_SUFFIX="0"
MEMORY="64G"
GPU_MEMORY="32G"
VERBOSE=0
RUN_COMMAND="/bin/bash"
N_GPUS_SET=1
NODE_POOLS=""
shift 1

while getopts m:l:c:g:p:s:u:o:v opt; do
	case ${opt} in
		m)
			MEMORY=${OPTARG}
			;;
		l)
			CLUSTER=${OPTARG}
			;;
		c)
			N_CPUS=${OPTARG}
			;;
		g)
			number_re='^[0-9]+$'
			if [[ ${OPTARG} =~ $number_re ]] ; then
				N_GPUS=${OPTARG}
			else
				GPU_MEMORY=${OPTARG}
				N_GPUS_SET=0
			fi
			;;
		p)
			JOB_PREFIX=${OPTARG}
			;;
		s)
			JOB_SUFFIX=${OPTARG}
			;;
		u)
			GPU_MEMORY=${OPTARG}
			;;
		r)
			RUN_COMMAND=${OPTARG}
			;;
		o)
			NODE_POOLS=${OPTARG}
			;;
		v)
			VERBOSE=1
			;;
		?)
			echo unexpected option: ${opt}
			;;
	esac
done

JOB_SUFFIX=${CLUSTER}-${JOB_SUFFIX}
JOB_NAME=${JOB_PREFIX}-${JOB_SUFFIX}

if [ "$VERBOSE" == 1 ]; then
	echo main command: ${COMMAND}
	echo image: ${IMAGE}
	echo cluster: ${CLUSTER}
	echo job name: ${JOB_NAME}
	echo "--------------------------------"
	
	echo cpus: ${N_CPUS}
	echo memory: ${MEMORY}

 	if [ "$N_GPUS_SET" == 1 ]; then
		echo gpus: ${N_GPUS}
	else
		echo gpu memory: ${GPU_MEMORY}
	fi
	echo node pools: ${NODE_POOLS}
	echo "--------------------------------"
fi

if [ "$CLUSTER" == "ic" ]; then
    echo "using cluster: IC"
    runai config cluster ic-caas
else
    echo "using cluster: RCP"
    runai config cluster rcp-caas-prod
fi

echo "--------------------------------"

GPU_ARGS=""
if [ "$N_GPUS_SET" == 1 ]; then
	if [[ "$N_GPUS" > 0 ]]; then
		GPU_ARGS="--gpu $N_GPUS"
	fi
else
	GPU_ARGS="--gpu-memory $GPU_MEMORY"
fi

NODE_POOLS_ARGS=""
if [ "$NODE_POOLS" != "" ]; then
	NODE_POOLS_ARGS="--node-pools $NODE_POOLS"
fi

# Run this for train mode
if [ "$COMMAND" == "run" ]; then
	echo "Job [$JOB_NAME]"

	runai submit $JOB_NAME \
		-i $IMAGE \
		--cpu $N_CPUS \
		--cpu-limit $N_CPUS \
		--memory $MEMORY \
		--memory-limit $MEMORY \
		$GPU_ARGS \
		--pvc runai-nlp-ismayilz-nlpdata1:/mnt/nlpdata1 \
		--pvc runai-nlp-ismayilz-scratch:/mnt/scratch \
		--command -- bash entrypoint.sh
	exit 0
fi

# Run this for interactive mode
if [ "$COMMAND" == "run_bash" ]; then
	echo "Job [$JOB_NAME]"

	if [ "$CLUSTER" == "ic" ]; then
		# IC RunAI
		runai submit $JOB_NAME \
			-i $IMAGE \
			--cpu $N_CPUS \
			--cpu-limit $N_CPUS \
			--memory $MEMORY \
			--memory-limit $MEMORY \
			$GPU_ARGS \
			--pvc runai-nlp-ismayilz-nlpdata1:/mnt/nlpdata1 \
			--pvc runai-nlp-ismayilz-scratch:/mnt/scratch \
			--interactive \
			--attach \
			--command -- $RUN_COMMAND
		exit 0
	else
		# RCP RunAI
		runai submit $JOB_NAME \
			-i $IMAGE \
			--cpu $N_CPUS \
			--cpu-limit $N_CPUS \
			--memory $MEMORY \
			--memory-limit $MEMORY \
			$NODE_POOLS_ARGS \
			$GPU_ARGS \
			--pvc nlp-scratch:/mnt/scratch \
			--interactive \
			--attach \
			--command -- $RUN_COMMAND
		exit 0
	fi
fi

if [ "$COMMAND" == "log" ]; then
	runai logs $JOB_NAME -f
	exit 0
fi

if [ "$COMMAND" == "stat" ]; then
	runai describe job $JOB_NAME 
	exit 0
fi

if [ "$COMMAND" == "del" ]; then
	runai delete job $JOB_NAME
	exit 0
fi

if [ $? -eq 0 ]; then
	runai list job
fi
