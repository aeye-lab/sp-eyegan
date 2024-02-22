# reading comprehension
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name clrgaze --fine_tune 0
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name ekyt --fine_tune 0
python -m sp_eyegan.evaluate_downstream_task_reading_comprehension --encoder_name rf


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
python -m sp_eyegan.evaluate_downstream_task_gender_classification --gpu 0 --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_gender_classification --gpu 0 --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_gender_classification --gpu 0 --encoder_name rf

# adhd classification
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Despicable_Me --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_The_Present --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Fractals --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Diary_of_a_Wimpy_Kid_Trailer --encoder_name clrgaze
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Despicable_Me --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_The_Present --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Fractals --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Diary_of_a_Wimpy_Kid_Trailer --encoder_name ekyt
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Despicable_Me --encoder_name rf
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_The_Present --encoder_name rf
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Fractals --encoder_name rf
python -m sp_eyegan.evaluate_downstream_task_adhd_classification --gpu 0 --video Video_Diary_of_a_Wimpy_Kid_Trailer --encoder_name rf
