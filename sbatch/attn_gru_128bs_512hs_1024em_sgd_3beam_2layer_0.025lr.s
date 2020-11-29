#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,k80_4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ago265@nyu.edu
#SBATCH --job-name="testing"
#SBATCH --output=/scratch/ago265/nlp_project/outputs/%j.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=ago265
source activate nmt_env

# Set project working directory
PROJECT=/scratch/${NETID}/nlp_project

# Set arguments
STUDY_NAME=attn_gru_128bs_512hs_1024em_sgd_3beam_2layer_0.025lr
SAVE_DIR=${PROJECT}/saved_models
DATA_DIR=${PROJECT}/nmt-vi-en/data/interim/iwslt15-en-vn
PLOT_DIR=${PROJECT}/plots
BATCH_SIZE=128
LR=0.025
SEED=42
SOURCE_NAME='vi'
TARGET_NAME='en'
ENC_HIDDEN=512
ENC_EMB=1024
ENC_LAYERS=2
RNN_TYPE='gru'
DEC_EMB=512
DEC_HIDDEN=1024
DEC_LAYERS=2
EPOCHS=15
ATTN=TRUE
BEAM_SIZE=3
OPTIM=sgd


cd ${PROJECT}
python ./nmt-vi-en/scripts/train_attention.py \
	--experiment ${STUDY_NAME} \
	--save_dir ${SAVE_DIR} \
	--data_dir ${DATA_DIR} \
	--plots_dir ${PLOT_DIR} \
	--batch_size ${BATCH_SIZE} \
	--learning_rate ${LR} \
	--seed ${SEED} \
	--source_name ${SOURCE_NAME} \
	--target_name ${TARGET_NAME} \
	--enc_emb ${ENC_EMB} \
	--enc_hidden ${ENC_HIDDEN} \
	--enc_layers ${ENC_LAYERS} \
	--rnn_type ${RNN_TYPE} \
	--dec_emb ${DEC_EMB} \
	--dec_hidden ${DEC_HIDDEN} \
	--dec_layers ${DEC_LAYERS} \
	--epochs ${EPOCHS} \
	--beam_size ${BEAM_SIZE} \
	--optim ${OPTIM} \
	--attn
