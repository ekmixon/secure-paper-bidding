# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from utils import assign_IP, load_X_and_H_inv, load_y, load_preds, solve_ip_adv, get_global_variable
import argparse
from logger import set_logger
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')
parser.add_argument('--input_dir', default='./data/', type=str, help='output dir')
parser.add_argument('--lam', default=1e-4, type=float, help='L2 regularization')
parser.add_argument('--num_seeds_X', default=1, type=int, help='seed for hashing')
parser.add_argument('--subsample_max', default=60, type=int, help='max subsampled training labels')
parser.add_argument('--K', default=50, type=int, help='top K reviewers per paper considered for assignment')
parser.add_argument('--L', default=0,  type=int, help='the colluding size for detect evalaution')
parser.add_argument('--start_id', default=None, type=int, help='start paper id')
parser.add_argument('--end_id', default=None, type=int, help='end paper id')
args = parser.parse_args()

logger = set_logger("detect fpr results evaluation")
logger.info(args)

def eval_assignment(assignment, y, tpms_mat):
    pos_frac_intersection = ((y * assignment) > 0).sum() / float((assignment > 0).sum())
    average_bids_score = (y * assignment).sum() / float((assignment > 0).sum())
    tpms_average = (tpms_mat * assignment).sum() / float((assignment > 0).sum())
    tpms_max_average = (tpms_mat * assignment).max(1).mean()
    has_bids = ((y * assignment) > 0).sum(1) > 0
    frac_one_bid = (has_bids.sum() / float(len(has_bids)))
    return [pos_frac_intersection, average_bids_score, tpms_average, tpms_max_average]

num_paper, num_reviewer, _, _, hashed_ratio = get_global_variable()
X_seeds = range(args.num_seeds_X)
K = args.K
lam = args.lam
subsample_max = args.subsample_max
L = args.L
start_id = args.start_id
end_id = args.end_id

tpms = (
    torch.load(f"{args.input_dir}/raw_data/tensor_data.pl")["tpms"]
    .numpy()
    .reshape([num_reviewer, num_paper])
    .transpose()
)


X_csr_s = []
H_inv_s = []
y, y_train = load_y(hashed_ratio, logger, subsample_max=subsample_max)
preds_s = []
for seed in X_seeds:
    X_csr, H_inv = load_X_and_H_inv(hashed_ratio, seed, logger, lam)
    preds = load_preds(X_csr, y_train, H_inv, hashed_ratio, seed, logger, lam, subsample_max=subsample_max)
    X_csr_s.append(X_csr)
    H_inv_s.append(H_inv)
    preds_s.append(preds)
    del X_csr, H_inv, preds

ensemble_preds = np.add.reduce(preds_s)
prev_rank = np.argsort(-(ensemble_preds))

prev_rank_K = prev_rank[:, :K]
thr = 2
data = np.load(
    f"{args.output_dir}/detect_fpr/defense_fpr_collusion_{L}_top_{K}_num_seeds_{args.num_seeds_X}_lam_{lam}_subsample_max_{subsample_max}_start_id_{start_id}_end_id_{end_id}.npz"
)

advs_rank = data["advs_rank"]
print_log = [
    np.sum(advs_rank[:, :5] >= K) / advs_rank[:, :5].size,
    np.sum(advs_rank >= K) / advs_rank.size,
]

under_reviewed = np.sum(np.sum(advs_rank < K, axis=1) <= thr)
for i in range(num_paper):
    if np.sum(advs_rank[i] < K) < thr + 1:
        advs_rank[i] = np.argsort(np.argsort(advs_rank[i])) + (K - thr - 1)
assignment_detect = assign_IP(ensemble_preds,  K=K, advs_rank=advs_rank)
print_log += eval_assignment(assignment_detect, y, tpms)
print_log = ["{:.3f}".format(log) for log in print_log]

logger.info(
    f"top 5 fpr: {print_log[0]}, top 50 fpr: {print_log[1]}, Frac. of pos.: {print_log[2]}, Avg. bids score: {print_log[3]}, Avg. TPMS: {print_log[4]}, Avg. max. TPMS: {print_log[5]}, Under-reviewed: {under_reviewed}"
)
