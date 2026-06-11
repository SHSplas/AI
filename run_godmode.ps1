$ErrorActionPreference = 'Stop'
Set-Location 'c:\Users\HP\Desktop\New folder'
$py = 'C:\Users\HP\AppData\Local\Python\pythoncore-3.14-64\python.exe'

& $py -u train.py --run-steps 17000 --batch-size 4 --accum-steps 4 --lr 8e-5 --warmup-steps 750 --eval-every 200 --eval-batches 4 --save-every 100 --sample-every 400 --log-every 20 --threads 6 --interop-threads 1 --ckpt-path cpu_gpt_jarvis_godmode_l6_v2048.pth --best-path cpu_gpt_jarvis_godmode_l6_v2048_best.pth --metrics-csv cpu_gpt_jarvis_godmode_l6_v2048_metrics.csv

& $py -u train.py --run-steps 1000 --batch-size 4 --accum-steps 4 --lr 3e-5 --warmup-steps 100 --eval-every 200 --eval-batches 4 --save-every 100 --sample-every 400 --log-every 20 --threads 6 --interop-threads 1 --ckpt-path cpu_gpt_jarvis_godmode_l6_v2048.pth --best-path cpu_gpt_jarvis_godmode_l6_v2048_best.pth --metrics-csv cpu_gpt_jarvis_godmode_l6_v2048_metrics.csv
