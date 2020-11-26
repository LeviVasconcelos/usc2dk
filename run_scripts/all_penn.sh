python train_source_da.py --config config/h36m_penn/only_dial.yaml --tgt_dataset penn --log_dir visualizations/h36m_penn/dial

python train_source_da.py --config config/h36m_penn/dial_gan.yaml --tgt_dataset penn --log_dir visualizations/h36m_penn/dial/gan

python train_source_da.py --config config/h36m_penn/dial_gan_transforms.yaml --tgt_dataset penn --log_dir visualizations/h36m_penn/dial/gan/transforms
