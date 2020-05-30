
train_titan:
	python3 main.py --from-epoch -1 --epoch 8 --lr 2e-4 --batch-size 3 --num-workers 16 --gpu 0 --model-path trained_model/ 

train_1080:
	python3 main.py --from-epoch -1 --epoch 5 --lr 2e-4 --batch-size 1 --num-workers 10 --gpu 1 --model-path trained_model/

export_video:
	python3 export_video.py --model-path trained_model/epoch_0_loss_0.0062.pt --data-dir /media/disk2/aa/SeeInDark/ --save-path result/ --gpu 1



