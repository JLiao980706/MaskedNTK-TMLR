python resnet_cifar_experiments.py run -m $1 -p 10 -x 0.8 -n 1000 -g 20 -r 3 -v 5 -e 10 -l 0.16 -f loss_dynamic_8.json
python resnet_cifar_experiments.py run -m 500 -p 10 -x 0.5 -n 1000 -g 20 -r 10 -v 5 -e 10 -l 0.1 -f loss_dynamic_5.json
python resnet_cifar_experiments.py run -m 500 -p 10 -x 0.3 -n 1000 -g 20 -r 10 -v 5 -e 10 -l 0.06 -f loss_dynamic_3.json