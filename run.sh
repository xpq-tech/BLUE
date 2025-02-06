# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/alpha_llama3_zsre_seqen_2_layer_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/alpha_llama3_zsre_seqen_100.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/alpha_llama3_mcf_seqen_2_layer.log &

# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/alpha_llama3_mcf_seqen.log &




# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_llama3_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/AlphaEdit_blue_llama3_mcf_batch10000.log &

# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/AlphaEdit_llama3_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=200 \
#         --num_edits=200 \
#         --downstream_eval_steps=-1 > logs/AlphaEdit_blue_gptj_mcf_batch200.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1  > logs/AlphaEdit_gptj_mcf_batch10000.log &


# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_rect_blue_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_prune \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_prune_blue_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=7 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_prune \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_prune_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_blue_gptj_mcf_batch10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_llama3_blue_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_rect_blue_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_rect_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_rect_blue_gptj_mcf_batch10000.log &

# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_rect_gptj_mcf_batch10000.log &

# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/MEMIT_seq_gptj_mcf_blue.log &


# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=1 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/MEMIT_gpt2_mcf.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/MEMIT_seq_gptj_mcf_100_normal.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=PMET \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/PMET_llama3_zsre_10000.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=PMET \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/PMET_seq_gpt2xl_mcf_10000.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=PMET \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-zsre-batch-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/PMET_seq_gpt2xl_zsre_10000.log &
# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_llama3_mcf.log &

# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_llama3_mcf_blue.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_llama3_zsre.log &

# CUDA_VISIBLE_DEVICES=7 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_llama3_zsre_blue.log &


# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_llama3_mcf.log &


# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_llama3_mcf_blue.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=300 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_llama3_zsre_300.log &


# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=-1 > logs/MEMIT_rect_llama3_zsre_blue.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache> logs/MEMIT_rect_llama3_zsre_blue_batch10000.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=1 > logs/MEMIT_gpt2_mcf_batch_10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=1 > logs/MEMIT_gpt2_zsre_batch_10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=NSE \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/NSE_llama3_mcf_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=PMET_seq \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/PMET_seq_blue_llama3_mcf_100.log &

# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=PMET \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10 \
#         --num_edits=10 \
#         --downstream_eval_steps=-1 > logs/PMET_llama3_mcf_100.log &

CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate_batch  \
        --alg_name=PMET \
        --model_name=EleutherAI/gpt-j-6B \
        --model_path=../ptms/ \
        --hparams_fname=EleutherAI_gpt-j-6B-zsre-batch-blue.json \
        --ds_name=zsre \
        --dataset_size_limit=10000 \
        --num_edits=10000 \
        --downstream_eval_steps=-1 \
        --use_cache > logs/PMET_blue_gptj_zsre_10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=PMET_seq \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/PMET_seq_gpt2xl_mcf_100.log &

# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/MEMIT_blue_llama3_mcf_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_prune \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_prune_blue_llama3_zsre_batch_10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_prune \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_prune_llama3_zsre_batch_10000.log &
# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_prune \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_prune_gptj_mcf_batch_10000.log &

# CUDA_VISIBLE_DEVICES=7 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_prune \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache  > logs/MEMIT_prune_blue_gptj_mcf_batch_10000.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache  > logs/MEMIT_rect_blue_gptj_mcf_batch_10000.log &

# CUDA_VISIBLE_DEVICES=7 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/AlphaEdit_blue_gptj_mcf_batch10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/AlphaEdit_blue_gptj_mcf_batch10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/AlphaEdit_blue_gptj_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_blue_gpt2xl_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_rect \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_rect_blue_gpt2xl_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/AlphaEdit_blue_gpt2xl_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache> logs/MEMIT_rect_llama3_zsre_batch10000.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=zsre \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/AlphaEdit_gpt2xl_zsre_batch10000.log &
# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_llama3_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_llama3_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_prune \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_prune_blue_llama3_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_llama3_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_llama3_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_gptj_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_gptj_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_prune \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_prune_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_prune \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_prune_blue_gptj_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=5 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gptj_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5  > logs/MEMIT_seq_blue_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=7 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_gptj_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gptj_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=6 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gptj_mcf_seq_100.log &


# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=-1 > logs/MEMIT_seq_gpt2xl_zsre_batch_100.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=EleutherAI/gpt-j-6B \
#         --model_path=../ptms/ \
#         --hparams_fname=EleutherAI_gpt-j-6B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_gptj_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_llama3_mcf_seq_100.log &



# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=AlphaEdit \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/AlphaEdit_blue_gpt2xl_mcf_batch_10000.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_gpt2xl_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_gpt2xl_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gpt2xl_mcf_seq_100.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate  \
#         --alg_name=AlphaEdit \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/AlphaEdit_blue_gpt2xl_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT_rect \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/MEMIT_rect_gpt2xl_mcf_batch100000.log &


# CUDA_VISIBLE_DEVICES=2 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 \
#         --use_cache > logs/MEMIT_gpt2xl_mcf_batch100000.log &


# CUDA_VISIBLE_DEVICES=3 nohup python3 -m experiments.evaluate_batch  \
#         --alg_name=MEMIT \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl.json \
#         --ds_name=mcf \
#         --dataset_size_limit=10000 \
#         --num_edits=10000 \
#         --downstream_eval_steps=-1 > logs/MEMIT_gpt2xl_mcf_batch100000.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_seq \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_seq_blue_gpt2xl_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_prune \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_prune_blue_gpt2xl_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_prune \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=2000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_prune_blue_gpt2xl_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=7 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=zsre \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gpt2xl_zsre_seq_100.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 -m experiments.evaluate  \
#         --alg_name=MEMIT_rect \
#         --model_name=gpt2-xl \
#         --hparams_fname=gpt2-xl-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=3000 \
#         --num_edits=100 \
#         --downstream_eval_steps=5 > logs/MEMIT_rect_blue_gpt2xl_mcf_seq_100.log &


# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=NSE \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=100 \
#         --downstream_eval_steps=-1 > logs/NSE_llama3_mcf_batch_100.log &


# CUDA_VISIBLE_DEVICES=4 nohup python3 -m experiments.evaluate  \
#         --alg_name=NSE \
#         --model_name=Meta-Llama-3-8B-Instruct \
#         --hparams_fname=Llama3-8B-blue.json \
#         --ds_name=mcf \
#         --dataset_size_limit=100 \
#         --num_edits=1 \
#         --downstream_eval_steps=-1 > logs/NSE_blue_llama3_mcf_100.log &
