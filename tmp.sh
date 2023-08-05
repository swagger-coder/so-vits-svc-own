bash run_unsplit.sh zhibo01 0 8890 > log_zhibo01.txt 2>&1 &
bash run_unsplit.sh zhibo02 1 8891 > log_zhibo02.txt 2>&1 &
bash run_unsplit.sh zhibo03 2 8892 > log_zhibo03.txt 2>&1 &
bash run_unsplit.sh zhibo04 3 8893 > log_zhibo04.txt 2>&1 &
bash run_unsplit.sh zhibo05 4 8894 > log_zhibo05.txt 2>&1 &
bash run_unsplit.sh zhibo06 5 8895 > log_zhibo06.txt 2>&1 &
bash run_unsplit.sh zhibo07 6 8896 > log_zhibo07.txt 2>&1 &
bash run_unsplit.sh zhibo08 7 8897 > log_zhibo08.txt 2>&1 &




bash run_unsplit.sh zhibo09 4 8898 > log_zhibo09.txt 2>&1 &
bash run_unsplit.sh zhibo10 5 8899 > log_zhibo10.txt 2>&1 &
bash run_unsplit.sh zhibo11 6 8900 > log_zhibo11.txt 2>&1 &
bash run_unsplit.sh zhibo12 7 8901 > log_zhibo12.txt 2>&1 &


python compress_model.py -c="configs/config_base_210.json" -i="logs/base/G_411200.pth" -o="logs/base/release.pth"