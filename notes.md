
* A large batch size does help improve stability of the network as there is a smaller variance between iteration losses

* There is only one conv layer in my own defined ones as the gradients start diminishing when the network is too stretched

* The loss values were calculated in the using the normalised image coordinates as they help provide small gradient updates. When I used large values for loss as a result of calculating the loss using global image coordinates, the gradients exploded.

* To verify my setup, I try and get the model to overfit on the batch of 1, 8, 16 and 24 image datasets using a batch size of 8.

* Most of the learning should be done in the first iteration and first epoch. See training logs

* I trained the model for two days on a single GPU, roughly 1hr for 3 epochs. Training for 150 epochs equals 2days 3hrs
    * Based on the results, the model seems to be overfitting around the 18th epoch, I'd try adding data augmentation and maybe mix the train and test datasets as the train dataset is not big enough

* TODO: I need to show how i setup the google compute engine and how I trained it
    * Command used to train the model `nohup python -u ./training.py > training.log &`
* TODO: I need to show how i downloaded the model using the gcloud console's download button
    * GCP scp command `gcloud compute scp --zone "us-east1-b" yolov1-machine-vm:~/.bashrc ~/Documents/`