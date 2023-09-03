# reading comprehension
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name clrgaze --fine_tune 0
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name ekyt --fine_tune 0


# biometrics
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset judo --gpu 0 --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset judo --gpu 1 --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset gazebase --gpu 0 --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset gazebase --gpu 1 --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset judo --gpu 0 --encoder_name clrgaze --fine_tune 0
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset judo --gpu 1 --encoder_name ekyt --fine_tune 0
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset gazebase --gpu 0 --encoder_name clrgaze --fine_tune 0
python -m sp_eyegan.evaluate_downstream_task_biometric --dataset gazebase --gpu 1 --encoder_name ekyt --fine_tune 0

# gender classification
python -m sp_eyegan.evaluate_downstream_task_gender_classification --gpu 0 --encoder_name clrgaze --flag_redo
python -m sp_eyegan.evaluate_downstream_task_gender_classification --gpu 0 --encoder_name ekyt

# adhd classification
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --encoder_name ekyt
