python -m sp_eyegan.pretrain_constastive_learning --stimulus text -augmentation_mode random  -sd 0.1 -sd_factor 1.25 -encoder_name ekyt -GPU 0 --window_size 5000
python -m sp_eyegan.pretrain_constastive_learning --stimulus text -augmentation_mode random  -sd 0.1 -sd_factor 1.25 -encoder_name clrgaze -GPU 1 --window_size 5000
