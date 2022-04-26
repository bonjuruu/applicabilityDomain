#!/bin/bash

bash ./bash/advx/cleanup.sh
bash ./bash/advx/train_clf.sh
bash ./bash/advx/gen_advx.sh
bash ./bash/advx/eval_ad.sh