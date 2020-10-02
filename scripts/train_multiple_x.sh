################## WITH SOURCE MULTI VS SINGLE DISC ON PENN ###########################

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt penn --no_feature_align --log_dir visualizations/icpr/penn/with_source/multiscale_disc --epochs 50 --multiscale

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt penn --no_feature_align --log_dir visualizations/icpr/penn/with_source/singlescale_disc --epochs 50 

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt lsp --no_feature_align --log_dir visualizations/icpr/lsp/with_source/multiscale_disc --epochs 50 --multiscale

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt lsp --no_feature_align --log_dir visualizations/icpr/lsp/with_source/singlescale_disc --epochs 50 

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt mpii --no_feature_align --log_dir visualizations/icpr/mpii/with_source/multiscale_disc --epochs 50 --multiscale

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt mpii --no_feature_align --log_dir visualizations/icpr/mpii/with_source/singlescale_disc --epochs 50 


################### NO SOURCE ADDA ####################

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt penn --no_output_align --log_dir visualizations/icpr/penn/no_source/ADDA/ --epochs 50 --no_source
python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt lsp --no_output_align --log_dir visualizations/icpr/lsp/no_source/ADDA/ --epochs 50 --no_source
python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt mpii --no_output_align --log_dir visualizations/icpr/mpii/no_source/ADDA/ --epochs 50 --no_source

################## NO SOURCE OURS ###################


python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt penn --no_feature_align --log_dir visualizations/icpr/penn/no_source/multiscale_disc/ --epochs 50 --no_source --multiscale
python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt lsp --no_feature_align --log_dir visualizations/lsp/no_source/multiscale_disc/ --epochs 50 --no_source --multiscale
python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt mpii --no_feature_align --log_dir visualizations/mpii/no_source/multiscale_disc/ --epochs 50 --no_source --multiscale

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt penn --no_feature_align --log_dir visualizations/icpr/penn/no_source/singlescale_disc/ --epochs 50 --no_source

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt lsp --no_feature_align --log_dir visualizations/lsp/no_source/singlescale_disc/ --epochs 50 --no_source

python scripts/train_masked_discrim.py --config config/u2dkp.yaml --src_model visualizations/source/h36m/u2dkp_source\ 11-12-19\ 19\:09\:37/log_-checkpoint-50.pth.tar --tgt mpii --no_feature_align --log_dir visualizations/mpii/no_source/singlescale_disc/ --epochs 50 --no_source






