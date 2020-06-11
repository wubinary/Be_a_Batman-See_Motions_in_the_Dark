
###########################################
############## Training ###################

DATASET_PATH=/media/D/SeeInDark 
DATASET_PATH=/media/disk2/aa/SeeInDark 

download_dataset:
	wget https://storage.googleapis.com/isl-datasets/DRV/short1.zip -P $(DATASET_PATH)
	wget https://storage.googleapis.com/isl-datasets/DRV/short2.zip -P $(DATASET_PATH)
	wget https://storage.googleapis.com/isl-datasets/DRV/short3.zip -P $(DATASET_PATH)
	wget https://storage.googleapis.com/isl-datasets/DRV/short4.zip -P $(DATASET_PATH)
	wget https://storage.googleapis.com/isl-datasets/DRV/short5.zip -P $(DATASET_PATH)
	wget https://storage.googleapis.com/isl-datasets/DRV/long.zip-P $(DATASET_PATH)

train_titan:
	python3 main.py --from-epoch -1 --epoch 8 --lr 2e-4 --batch-size 4 --num-workers 16 --gpu 0 --data-path $(DATASET_PATH) --model-path trained_model/ 

train_1080:
	python3 main.py --from-epoch -1 --epoch 3 --lr 2e-4 --batch-size 1 --num-workers 10 --gpu 1 --data-path $(DATASET_PATH) --model-path trained_model/

train_1080ti: 
	python3 main.py --from-epoch -1 --epoch 3 --lr 2e-4 --batch-size 2 --num-workers 10 --gpu 0 --data-path $(DATASET_PATH) --model-path trained_model/
	


###########################################
############### Testing ###################

MODEL=trained_model/epoch_2_loss_0.0059.pt 

export_video:
	python3 export_video.py --model-path $(MODEL) --data-dir $(DATASET_PATH) --gpu 0


 
