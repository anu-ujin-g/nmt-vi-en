#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,k80_4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ltk224@nyu.edu
#SBATCH --job-name="transformer_test"
#SBATCH --output=/scratch/ltk224/nlp_project/outputs/%j.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=ltk224
source activate nmt_env

# Set project working directory
PROJECT=/scratch/${NETID}/nlp_project

# Set arguments
STUDY_NAME=nmt_attn_1gpu 
SAVE_DIR=${PROJECT}/saved_models
DATA_DIR=${PROJECT}/vietnamese-chatbot-2/data2/iwslt-vi-en
BATCH_SIZE=128
LR=0.25
SEED=42
SOURCE_NAME='vi'
TARGET_NAME='en'
ENC_EMB=512
ENC_HIDDEN=256
ENC_LAYERS=3
RNN_TYPE='lstm'
DEC_EMB=512
DEC_HIDDEN=512
DEC_LAYERS=1
EPOCHS=15
ENC_PTH_NAME='enc_enchid256_enclay3'
DEC_PTH_NAME='dec_enchid256_enclay3'

# ENCODER_ATTN=TRUE
# SELF_ATTN=FALSE

# LONGEST_LABEL=1
# GRADIENT_CLIP=0.3

cd ${PROJECT}
python ./vietnamese-chatbot-2/scripts/train_attention.py \
	--experiment ${STUDY_NAME} \
	--save_dir ${SAVE_DIR} \
	--data_dir ${DATA_DIR} \
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
	--enc_pth_name ${ENC_PTH_NAME} \
	--dec_pth_name ${DEC_PTH_NAME}
	# --encoder_attention \
	# --self_attention
	# --longest_label ${LONGEST_LABEL} \
	# --gradient_clip ${GRADIENT_CLIP} \