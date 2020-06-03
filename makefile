
dataset_path=/media/D/SeeInDark
download_dataset:
	wget https://storage.googleapis.com/isl-datasets/DRV/short1.zip -P $(dataset_path)
	wget https://storage.googleapis.com/isl-datasets/DRV/short2.zip -P $(dataset_path)
	wget https://storage.googleapis.com/isl-datasets/DRV/short3.zip -P $(dataset_path)
	wget https://storage.googleapis.com/isl-datasets/DRV/short4.zip -P $(dataset_path)
	wget https://storage.googleapis.com/isl-datasets/DRV/short5.zip -P $(dataset_path)
	wget https://storage.googleapis.com/isl-datasets/DRV/long.zip-P $(dataset_path)

train_titan:
	python3 main.py --from-epoch -1 --epoch 3 --lr 2e-4 --batch-size 5 --num-workers 16 --gpu 0 --data-path $(dataset_path) --model-path trained_model/ 

train_1080:
	python3 main.py --from-epoch -1 --epoch 3 --lr 2e-4 --batch-size 1 --num-workers 10 --gpu 1 --data-path $(dataset_path) --model-path trained_model/

train_1080ti: 
	python3 main.py --from-epoch -1 --epoch 3 --lr 2e-4 --batch-size 2 --num-workers 10 --gpu 0 --data-path $(dataset_path) --model-path trained_model/



result_path=result/
model=trained_model/epoch_0_loss_0.0083.pt
export_video:
	python3 export_video.py --model-path $(model) --data-dir $(dataset_path) --save-path $(result_path) --gpu 1


 
