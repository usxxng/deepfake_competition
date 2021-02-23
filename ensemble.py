import pandas as pd
import numpy as np
from glob import glob
import argparse
import os
from sklearn.metrics import roc_auc_score


'''
# 모델 앙상블

총 18개 학습한 모델을 predict를 통해 평가한 결과가 저장된 csv파일
      ./subs/*.csv
를 읽어온 뒤, 모든 값의 평균을 내서 결과를 평가함

# When ensembling different folds, or different models,
# we first rank all the probabilities of each model/fold,
# to ensure they are evenly distributed.
# In pandas, it can be done by df['pred'] = df['pred'].rank(extract)

'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, default='sampled_face/')
    parser.add_argument('--ensemble-name', type=str, default='final_sub')
    args, _ = parser.parse_known_args()
    return args


def get_median_value(subs):
    # probs_sorted = np.sort(sub_probs)
    #
    # prob_len = len(probs_sorted)
    return np.median(subs)

def normalize_median_hist(subs):
    # probs_sorted = np.sort(sub_probs)
    #
    # prob_len = len(probs_sorted)
    return np.median(subs)



if __name__ == '__main__':
    args = parse_args()

    # 폴더에서 csv읽어오기
    subs = [pd.read_csv(csv) for csv in sorted(glob(os.path.join(args.sub_dir, '*csv')))]
    sub_probs = [sub.target for sub in subs]

    # 앙상블을 위한 균등 가중치
    ensem_number = len(sub_probs)
    wts = [1/ensem_number]*ensem_number




    # 가중치 반영하여 결과 평균내기
    MEDIANS = [get_median_value(subs) for subs in sub_probs]
    MED_PROBS = np.zeros(4100)
    #MED_PROBS = [(subs[i].target > MEDIANS[i]) * 1 for i, subs in enumerate(sub_probs)]


    for i in range(ensem_number):
        MED_PROBS += ((sub_probs[i] > MEDIANS[i]) * 1) * wts[i]


    #PROBS = np.sum([wts[i]*sub_probs[i] for i in range(len(wts))], axis=0)


    #PROBS = (np.sum([wts[i]*sub_probs[i] for i in range(len(wts))], axis=0)>0.5)*1
    PROBS = (MED_PROBS > 0.5) * 1#(np.sum([wts[i]*sub_probs[i] for i in range(len(wts))], axis=0)>0.5)*1


    print(np.mean(PROBS))


    # test 정답이 있는 경우 읽어온다
    df_test = pd.read_csv(os.path.join(args.data_dir, args.data_folder, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(args.data_dir, f'{args.data_folder}test', x))

    df_test['path'] = df_test['image_name'].apply(lambda x: os.path.join(f'leaderboard/', x))


    # test 정답이 있는 경우 정확도 평가를 진행
    if 'target' in df_test.columns:
        diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_test.target.unique()))}
        df_test['target'] = df_test['target'].map(diagnosis2idx)

        # CSV 기준 정상은 0, 비정상은 1 여기서는 stone 데이터를 타겟으로 삼음
        target_idx = diagnosis2idx[1]
        TARGETS = df_test['target']

        # 정확도 평가
        df_test['target'] = PROBS
        acc = (np.round(PROBS) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS)

        # 앙상블 최종 결과를 저장함
        # 앙상블 최종 결과를 저장함
        df_sub = subs[0]
        df_sub['path'] = df_test['path']
        df_sub['y'] = PROBS
        df_sub = df_sub.drop(['target', 'image_name'], axis=1)
        df_sub.to_csv(f"{args.ensemble_name}.csv", index=False)
        df_sub.to_csv(f'./{args.ensemble_name}_{acc:.2f}_{auc:.4f}.csv', index=False)

    else:

        # Test에 대한 정답이 없는 경우
        # 앙상블 최종 결과를 저장함
        df_sub = subs[0]
        df_sub['path'] = df_test['path']
        df_sub['y'] = PROBS
        df_sub = df_sub.drop(['target', 'image_name'], axis=1)
        df_sub.to_csv(f"{args.ensemble_name}.csv", index=False)




